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


struct Opts {
	const char* compiler_option;
	cl_device_type device_type;
	unsigned int array_size1;
	unsigned int array_size2;
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


struct point {
	float startx;
	float starty;
	float midx;
	float midy;
	float endx;
	float endy;
};


struct point* output=NULL;
struct point* operand1=NULL;
struct point* operand2=NULL;


const char* kernelSourceCode=KERNEL(
	typedef struct {
		float startx;
		float starty;
		float midx;
		float midy;
		float endx;
		float endy;
	}point;

	__kernel void computemap(__global point* operand1,__global point* operand2,__global int stage,__global point* output){
		size_t size=get_global_size(0);
		size_t col_idx=get_global_id(1);
		size_t row_idx=get_global_id(0);
		size_t col_offset=get_global_offset(1);
		size_t row_offset=get_global_offset(0);
		size_t index=(row_idx-row_offset)*size+(col_idx-col_offset);

		output[index].startx=col_idx;
		output[index].starty=row_idx;
		output[index].endx=col_offset;
		output[index].endy=row_offset;
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


static void DumpBuffer(const Opts* opts, const point* buffer){
	for (size_t ii = 0; ii < opts->array_size1*opts->array_size2; ii++) {
		if (ii % 10 == 0) {
			printf("\n%4d:", ii);
		}
		printf("(%.0f,%.0f) ", buffer[ii].startx,buffer[ii].starty);
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
	cl_int status;
	cl_program program;
	const char *source = FileToString(opts);
	size_t sourceSize[] = {strlen(source)};
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
	size_t global_work_size[2]={opts->array_size1,opts->array_size2};
	if (!LoadKernel(opts,&kernel,goo)) {
		return 0;
	}
	cl_mem input1=clCreateBuffer(goo->ctx,CL_MEM_COPY_HOST_PTR,opts->array_size1*sizeof(struct point),operand1,&status);
	if(status!=CL_SUCCESS){
		printf("Error: clCreateBuffer 1.\n");
		return EXIT_FAILURE;
	}
	cl_mem input2=clCreateBuffer(goo->ctx,CL_MEM_COPY_HOST_PTR,opts->array_size2*sizeof(struct point),operand2,&status);
	if(status!=CL_SUCCESS){
		printf("Error: clCreateBuffer 2.\n");
		return EXIT_FAILURE;
	}
	cl_mem outputBuffer=clCreateBuffer(goo->ctx,CL_MEM_WRITE_ONLY,opts->array_size1*opts->array_size2*sizeof(struct point),NULL,&status);
	if(status!=CL_SUCCESS){
		printf("Error: clCreateBuffer 3.\n");
		return EXIT_FAILURE;
	}
	status=clSetKernelArg(kernel,0,sizeof(cl_mem),(void*)&input1);
	if(status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 0.\n");
		return EXIT_FAILURE;
	}
	status=clSetKernelArg(kernel,1,sizeof(cl_uint),(void*)&(opts->array_size1));
	if (status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 1.\n");
		return EXIT_FAILURE;
	}
	status=clSetKernelArg(kernel,2,sizeof(cl_mem),(void*)&input2);
	if(status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 2.\n");
		return EXIT_FAILURE;
	}
	status=clSetKernelArg(kernel,3,sizeof(cl_uint),(void*)&(opts->array_size2));
	if (status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 3.\n");
		return EXIT_FAILURE;
	}
	status=clSetKernelArg(kernel,5,sizeof(cl_mem),(void*)&outputBuffer);
	if(status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 5.\n");
		return EXIT_FAILURE;
	}
	cl_uint stage=0;
	//for (int stage=0;stage<opts->array_size2;stage++){
		status=clSetKernelArg(kernel,4,sizeof(cl_uint),(void*)&stage);
		if(status!=CL_SUCCESS){
			printf("Error: clSetKernelArg 4.\n");
			return EXIT_FAILURE;
		}
	//}
	if (status!=CL_SUCCESS){
		printf("Error: clSetKernelArg 3.\n");
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
	status=clEnqueueReadBuffer(goo->queue,outputBuffer,CL_TRUE,0,opts->array_size1*opts->array_size2*sizeof(struct point),output,0,NULL,NULL);
	if (status!=CL_SUCCESS){
		printf("Error: clEnqueueReadBuffer.\n");
		return EXIT_FAILURE;
	}
	if (!opts->quiet){
		DumpBuffer(opts,output);
	}
	clReleaseEvent(event);
	status=clReleaseKernel(kernel);
	status=clReleaseMemObject(input1);
	status=clReleaseMemObject(input2);
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
//  bool quiet;
//	int timing;
//};
int main(int argc, char * argv[])
{
	CLGoo goo;
	
	//Opts opts = {"",CL_DEVICE_TYPE_CPU,1000,"computemap","",false,false,true};
	Opts opts = {"-DARRAY_SIZE=256*256",CL_DEVICE_TYPE_GPU,256,256,"computemap","convolution.cl",true,false,true};
	
	operand1=(struct point*)malloc(sizeof(struct point)*opts.array_size1);
	operand2=(struct point*)malloc(sizeof(struct point)*opts.array_size2);
	output=(struct point*)malloc(sizeof(struct point)*opts.array_size1*opts.array_size2);

	for (unsigned int i=0;i<opts.array_size1;i++){
		operand1[i].endx=i*1.0f;
		operand1[i].endy=i*1.0f;
		operand1[i].midx=i*1.0f;
		operand1[i].midy=i*1.0f;
		operand1[i].startx=i*1.0f;
		operand1[i].starty=i*1.0f;
	}
	for (unsigned int i=0;i<opts.array_size2;i++){
		operand2[i].endx=i*1.0f;
		operand2[i].endy=i*1.0f;
		operand2[i].midx=i*1.0f;
		operand2[i].midy=i*1.0f;
		operand2[i].startx=i*1.0f;
		operand2[i].starty=i*1.0f;
	}
	
	if (!InitializeCL(&goo, &opts)) {
		exit(1);
	}
	
	if (!RunTest(&goo, &opts)) {
		exit(1);
	}
	
	CleanupCL(&goo);
	
	free(output);
	free(operand1);
	free(operand2);
	
	return 0;
}