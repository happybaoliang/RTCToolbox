typedef struct {
	float startx;
	float starty;
	float midx;
	float midy;
	float endx;
	float endy;
}point;

__kernel void computemap(__global point *operand1,const int num1,__global point *operand2,const int num2,const int stage,__global point *output){
		size_t size=get_global_size(0);
		size_t col_idx=get_global_id(1);
		size_t row_idx=get_global_id(0);
		size_t col_offset=get_global_offset(1);
		size_t row_offset=get_global_offset(0);
		size_t index=size*(row_idx-row_offset)+(col_idx-col_offset);
}
