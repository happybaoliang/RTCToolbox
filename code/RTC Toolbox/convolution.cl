
__kernel void computemap(__global uint* startx1,__global uint* endx1,__global uint* y1,__global uint* startx2,
											__global uint* endx2,__global uint* y2,__global uint* map){

	__local volatile uint localmap[2*ARRAY_SIZE];

	size_t xId=get_global_id(0);
	size_t yId=get_global_id(1);

	localmap[2*xId]=INFINITY;
	localmap[2*xId+1]=INFINITY;
	
	barrier(CLK_LOCAL_MEM_FENCE);

	uint val=y1[xId]+y2[yId];
	uint endx=endx1[xId]+endx2[yId];
	uint startx=startx1[xId]+startx2[yId];

	for (int i=startx;i<=endx;i++)
		atomic_min(localmap+i,val);
	
	barrier(CLK_LOCAL_MEM_FENCE);

	map[2*xId]=localmap[2*xId];
	map[2*xId+1]=localmap[2*xId+1];
}