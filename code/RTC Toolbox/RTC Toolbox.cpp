#include <CL/cl.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <time.h>

using namespace std;


#define KERNEL(...)	#__VA_ARGS__
#define Warning(...)    fprintf(stderr, __VA_ARGS__)


unsigned int* output;
unsigned int* cendx1;
unsigned int* cendx2;
unsigned int* cvalue1;
unsigned int* cvalue2;
unsigned int* cstartx1;
unsigned int* cstartx2;


struct Opts {
	const char* compiler_option;
	cl_device_type device_type;
	unsigned int array_size;
	const char *kernelName;
	const char *fileName;
	bool fromfile;
	bool quiet;
	int timing;
};


struct CLGoo {
   cl_context ctx;
   cl_device_id *devices;
   cl_uint numDevices;
   cl_command_queue queue;
};


const char* kernelSourceCode=KERNEL(
	__kernel void computemap(__global uint* startx1,__global uint* endx1,__global uint* y1,__global uint* startx2,
		__global uint* endx2,__global uint* y2,__global uint* map){
	size_t xId=get_global_id(0);
	size_t yId=get_global_id(1);

	uint val=y1[xId]+y2[yId];
	uint endx=endx1[xId]+endx2[yId];
	uint startx=startx1[xId]+startx2[yId];

	for (int i=startx;i<=endx;i++)
		atomic_min(map+i,val);
}
);


static struct { cl_int code; const char *msg; } error_table[] = {
      { CL_SUCCESS, "success" },
      { CL_DEVICE_NOT_FOUND, "device not found", },
      { CL_DEVICE_NOT_AVAILABLE, "device not available", },
      { CL_COMPILER_NOT_AVAILABLE, "compiler not available", },
      { CL_MEM_OBJECT_ALLOCATION_FAILURE, "mem object allocation failure", },
      { CL_OUT_OF_RESOURCES, "out of resources", },
      { CL_OUT_OF_HOST_MEMORY, "out of host memory", },
      { CL_PROFILING_INFO_NOT_AVAILABLE, "profiling not available", },
      { CL_MEM_COPY_OVERLAP, "memcopy overlaps", },
      { CL_IMAGE_FORMAT_MISMATCH, "image format mismatch", },
      { CL_IMAGE_FORMAT_NOT_SUPPORTED, "image format not supported", },
      { CL_BUILD_PROGRAM_FAILURE, "build program failed", },
      { CL_MAP_FAILURE, "map failed", },
      { CL_INVALID_VALUE, "invalid value", },
      { CL_INVALID_DEVICE_TYPE, "invalid device type", },
      { 0, NULL },
};


static void DumpBuffer(const Opts* opts, const unsigned int *buffer){
	for (size_t ii = 0; ii < 2*opts->array_size; ii++) {
		if (ii % 10 == 0) {
			printf("\n%5d:", ii);
		}
		printf(" %5d", buffer[ii]);
	}
	printf("\n");
}


static const char * StrCLError(cl_int status) {
	static char unknown[25];
	for (int ii = 0; error_table[ii].msg != NULL; ii++) {
		if (error_table[ii].code == status) {
			return error_table[ii].msg;
		}
	}

	sprintf_s(unknown, "unknown error %d\n", status);
	return unknown;
}


static void CL_CALLBACK HandleCLError(const char *errInfo, const void *opaque, size_t opaqueSize, void *userData) {
	Warning("Unexpected OpenCL error: %s\n", errInfo);
	if (opaqueSize > 0) {
		Warning("  %d bytes of vendor data.\n", opaqueSize);
		for (size_t ii = 0; ii < opaqueSize; ii++) {
			char c = ((const char *) opaque)[ii];
			if (ii % 10 == 0) {
				Warning("\n   %3d:", ii);
			}
			Warning(" 0x%02x %c", c, isprint(c) ? c : '.');
		}
		Warning("\n");
	}
}


static int InitializeCL(CLGoo *goo, const Opts *opts){
	cl_command_queue_properties queueProps;
	size_t deviceListSize;
	size_t platformNum;
	cl_platform_id* platforms;
	cl_int status;

	memset(goo, 0, sizeof(*goo));

	status=clGetPlatformIDs(0,NULL,&platformNum);
	if(status!=CL_SUCCESS){
		Warning("unable to get platform number: %d\n",status);
		return -1;
	}

	platforms=(cl_platform_id*)malloc(platformNum*sizeof(cl_platform_id));

	status=clGetPlatformIDs(platformNum,platforms,NULL);
	if (status!=CL_SUCCESS){
		Warning("unable to get platform: %d\n",status);
		return -1;
	}

	cl_context_properties prop[] = {CL_CONTEXT_PLATFORM,(cl_context_properties)platforms[0],0};

	goo->ctx = clCreateContextFromType(prop, opts->device_type,HandleCLError, NULL, &status);
	if (status != CL_SUCCESS) {
		Warning("clCreateContextFromType failed: %s\n", StrCLError(status));
		return 0;
	}

	status = clGetContextInfo(goo->ctx,CL_CONTEXT_DEVICES, 0, NULL, &deviceListSize);
	if (status != CL_SUCCESS) {
		Warning("clGetContextInfo failed: %s\n", StrCLError(status));
		return 0;
	}
	goo->numDevices = deviceListSize / sizeof(cl_device_id);

	if ((goo->devices = (cl_device_id *) malloc(deviceListSize)) == NULL) {
		Warning("Failed to allocate memory (deviceList).\n");
		return 0;
	}

	status = clGetContextInfo(goo->ctx, CL_CONTEXT_DEVICES, deviceListSize,goo->devices, NULL);
	if (status != CL_SUCCESS) {
		Warning("clGetGetContextInfo failed: %s\n", StrCLError(status));
		return 0;
	}

	queueProps = opts->timing ? CL_QUEUE_PROFILING_ENABLE : 0;
	goo->queue = clCreateCommandQueue(goo->ctx,goo->devices[0], queueProps, &status);
	if (status != CL_SUCCESS) {
		Warning("clCreateCommandQueue failed: %s\n", StrCLError(status));
		return 0;
	}

	return 1;
}


static const char * FileToString(const Opts* opts) {
	if (opts->fromfile){
		char *contents;
		size_t program_size;
		FILE* program_handle;
		if (fopen_s(&program_handle,opts->fileName, "rb")!=0){
			Warning("Couldn't find the program file");
			return NULL;;
		}else{
			fseek(program_handle, 0, SEEK_END);
			program_size = ftell(program_handle);
			rewind(program_handle);
			contents = (char*)malloc(program_size+1);
			contents[program_size] = '\0';
			fread(contents, sizeof(char),program_size, program_handle);
			fclose(program_handle);
			return contents;
		}
	}else{
		return kernelSourceCode;
   }
}


static int LoadKernel(const Opts* opts, cl_kernel *kernel, CLGoo *goo) {
	const char *source = FileToString(opts);
	size_t sourceSize[] = {strlen(source)};
	cl_program program;
	cl_int status;

	program = clCreateProgramWithSource(goo->ctx, 1,&source, sourceSize, &status);
	if (status != CL_SUCCESS) {
		Warning("clCreateProgramWithSource failed: %s", StrCLError(status));
		return 0;
	}

	status = clBuildProgram(program, 1, goo->devices, opts->compiler_option, NULL, NULL);
	if (status != CL_SUCCESS) {
		size_t log_size;
		char* program_log;
		Warning("clBuildProgram failed: %s\n", StrCLError(status));
		clGetProgramBuildInfo(program, goo->devices[0], CL_PROGRAM_BUILD_LOG,0, NULL, &log_size);
		program_log = (char*) calloc(log_size+1, sizeof(char));
		clGetProgramBuildInfo(program, goo->devices[0], CL_PROGRAM_BUILD_LOG,log_size+1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		return 0;
	}

	*kernel = clCreateKernel(program, opts->kernelName, &status);
	if (status != CL_SUCCESS) {
		Warning("clCreateKernel(%s) failed: %s\n",opts->kernelName, StrCLError(status));
		return 0;
	}

	if (clReleaseProgram(program) != CL_SUCCESS) {
		Warning("clReleaseProgram() failed.  Bummer.\n");
	}

	return 1;
}


static int RunTest(CLGoo *goo, const Opts *opts) {
	cl_int status;
	cl_event event;
	cl_kernel kernel;
	size_t global_work_size[2]={opts->array_size,opts->array_size};

	if (!LoadKernel(opts,&kernel,goo)) {
		return 0;
	}

	cl_mem endx1=clCreateBuffer(goo->ctx,CL_MEM_COPY_HOST_PTR,opts->array_size*sizeof(unsigned int),cendx1,&status);
	if(status!=CL_SUCCESS){
		printf("Error: clCreateBuffer 1.\n");
		return EXIT_FAILURE;
	}
	cl_mem endx2=clCreateBuffer(goo->ctx,CL_MEM_COPY_HOST_PTR,opts->array_size*sizeof(unsigned int),cendx2,&status);
	if(status!=CL_SUCCESS){
		printf("Error: clCreateBuffer 2.\n");
		return EXIT_FAILURE;
	}
	cl_mem value1=clCreateBuffer(goo->ctx,CL_MEM_COPY_HOST_PTR,opts->array_size*sizeof(unsigned int),cvalue1,&status);
	if(status!=CL_SUCCESS){
		printf("Error: clCreateBuffer 3.\n");
		return EXIT_FAILURE;
	}
	cl_mem value2=clCreateBuffer(goo->ctx,CL_MEM_COPY_HOST_PTR,opts->array_size*sizeof(unsigned int),cvalue2,&status);
	if(status!=CL_SUCCESS){
		printf("Error: clCreateBuffer 4.\n");
		return EXIT_FAILURE;
	}
	cl_mem startx1=clCreateBuffer(goo->ctx,CL_MEM_COPY_HOST_PTR,opts->array_size*sizeof(unsigned int),cstartx1,&status);
	if(status!=CL_SUCCESS){
		printf("Error: clCreateBuffer 5.\n");
		return EXIT_FAILURE;
	}
	cl_mem startx2=clCreateBuffer(goo->ctx,CL_MEM_COPY_HOST_PTR,opts->array_size*sizeof(unsigned int),cstartx2,&status);
	if(status!=CL_SUCCESS){
		printf("Error: clCreateBuffer 6.\n");
		return EXIT_FAILURE;
	}
	cl_mem outputBuffer=clCreateBuffer(goo->ctx,CL_MEM_READ_ONLY,2*opts->array_size*sizeof(unsigned int),NULL,&status);
	if(status!=CL_SUCCESS){
		printf("Error: clCreateBuffer 7.\n");
		return EXIT_FAILURE;
	}

	status=clSetKernelArg(kernel,0,sizeof(cl_mem),(void*)&startx1);
	if(status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 1.\n");
		return EXIT_FAILURE;
	}
	status=clSetKernelArg(kernel,1,sizeof(cl_mem),(void*)&endx1);
	if(status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 2.\n");
		return EXIT_FAILURE;
	}
	status=clSetKernelArg(kernel,2,sizeof(cl_mem),(void*)&value1);
	if(status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 3.\n");
		return EXIT_FAILURE;
	}
	status=clSetKernelArg(kernel,3,sizeof(cl_mem),(void*)&startx2);
	if(status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 4.\n");
		return EXIT_FAILURE;
	}
	status=clSetKernelArg(kernel,4,sizeof(cl_mem),(void*)&endx2);
	if(status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 5.\n");
		return EXIT_FAILURE;
	}
	status=clSetKernelArg(kernel,5,sizeof(cl_mem),(void*)&value2);
	if(status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 6.\n");
		return EXIT_FAILURE;
	}
	status=clSetKernelArg(kernel,6,sizeof(cl_mem),(void*)&outputBuffer);
	if(status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 7.\n");
		return EXIT_FAILURE;
	}

	status = clEnqueueNDRangeKernel(goo->queue, kernel, 2, NULL,global_work_size, NULL, 0, NULL, &event);
	if (status != CL_SUCCESS) {
		Warning("Failed to launch kernel: %s\n", StrCLError(status));
		return 0;
	}

	if ((status = clWaitForEvents(1, &event)) != CL_SUCCESS) {
		Warning("clWaitForEvents() failed: %s\n", StrCLError(status));
		clFinish(goo->queue); 
	}

	if (opts->timing) {
		double total;
		long long start, end;

		status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,sizeof(start), &start, NULL);
		if (status != CL_SUCCESS) {
			Warning("clGetEventProfilingInfo(COMMAND_START) failed: %s\n",StrCLError(status));
			start = 0;
		}
		status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,sizeof(end), &end, NULL);
		if (status != CL_SUCCESS) {
			Warning("clGetEventProfilingInfo(COMMAND_END) failed: %s\n",StrCLError(status));
			end = 0;
		}

		total = (double)(end - start) / 1e6;
		printf("Profiling: Total kernel time was %5.2f msecs.\n", total);
	}

	status=clEnqueueReadBuffer(goo->queue,outputBuffer,CL_TRUE,0,2*opts->array_size,output,0,NULL,NULL);
	if (status!=CL_SUCCESS){
		printf("Error: clEnqueueReadBuffer.\n");
		return EXIT_FAILURE;
	}

	if (!opts->quiet){
		DumpBuffer(opts,output);
	}

	clReleaseEvent(event);
	status=clReleaseKernel(kernel);
	status=clReleaseMemObject(endx1);
	status=clReleaseMemObject(endx2);
	status=clReleaseMemObject(value1);
	status=clReleaseMemObject(value2);
	status=clReleaseMemObject(startx1);
	status=clReleaseMemObject(startx2);
	status=clReleaseMemObject(outputBuffer);

	return 1;
}


static void CleanupCL(CLGoo *goo) {
	cl_int status;

	if (goo->queue != NULL) {
		if ((status = clReleaseCommandQueue(goo->queue)) != CL_SUCCESS) {
			Warning("clReleaseCommandQueue failed: %s", StrCLError(status));
		}
		goo->queue = NULL;
	}

	free(goo->devices);
	goo->numDevices = 0;

	if (goo->ctx != NULL) {
		if ((status = clReleaseContext(goo->ctx)) != CL_SUCCESS) {
			Warning("clReleaseContext failed: %s", StrCLError(status));
		}
		goo->ctx = NULL;
	}
	return;
}


//struct Opts {
//	const char* compiler_option;
//	cl_device_type device_type;
//	unsigned int array_size;
//	const char *kernelName;
//	const char *fileName;
//	bool fromfile;
//	int timing;
//};
int main(int argc, char * argv[])
{
	CLGoo goo;

	//Opts opts = {"-DARRAY_SIZE=1000",CL_DEVICE_TYPE_CPU,1000,"computemap","convolution.cl",false,true,true};
	Opts opts = {"-DARRAY_SIZE=100",CL_DEVICE_TYPE_GPU,100,"computemap","convolution.cl",true,false,true};

	cendx1=(unsigned int*)malloc(sizeof(unsigned int)*opts.array_size);
	cendx2=(unsigned int*)malloc(sizeof(unsigned int)*opts.array_size);
	cvalue1=(unsigned int*)malloc(sizeof(unsigned int)*opts.array_size);
	cvalue2=(unsigned int*)malloc(sizeof(unsigned int)*opts.array_size);
	cstartx1=(unsigned int*)malloc(sizeof(unsigned int)*opts.array_size);
	cstartx2=(unsigned int*)malloc(sizeof(unsigned int)*opts.array_size);
	output=(unsigned int*)malloc(2*sizeof(unsigned int)*opts.array_size);

	for (unsigned int i=0;i<opts.array_size;i++){
		cvalue1[i]=i;
		cvalue2[i]=i;
		cstartx1[i]=i;
		cstartx2[i]=i;
		cendx1[i]=i+1;
		cendx2[i]=i+1;
	}
	
	if (!InitializeCL(&goo, &opts)) {
		exit(1);
	}

	if (!RunTest(&goo, &opts)) {
		exit(1);
	}

	CleanupCL(&goo);

	free(output);
	free(cendx1);
	free(cendx2);
	free(cvalue1);
	free(cvalue2);
	free(cstartx1);
	free(cstartx2);

	return 0;
}