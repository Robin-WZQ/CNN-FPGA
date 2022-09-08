// 权重读取

module weightMemory(clk,address,weights);

// 注意这都是默认参数，输入的时候可以改变
parameter DATA_WIDTH = 32; // 数据位宽
parameter INPUT_NODES = 100; // 输入节点数目
parameter OUTPUT_NODES = 32; // 输出节点数目
parameter file = "E:/Parameters With No Seperators/weights1_IEEE.txt"; //输入权重文档

localparam TOTAL_WEIGHT_SIZE = INPUT_NODES * OUTPUT_NODES; // 总共有x个权重

input clk; //时钟信号
input [8:0] address; // 位宽为8，深度为1
output reg [DATA_WIDTH*OUTPUT_NODES-1:0] weights; // 存储权重，位宽为x，深度为1

reg [DATA_WIDTH-1:0] memory [0:TOTAL_WEIGHT_SIZE-1]; // 32位位宽，存储权重

integer i;

always @ (posedge clk) begin	
	if (address > INPUT_NODES-1 || address < 0) begin
		weights = 0;
	end else begin
		for (i = 0; i < OUTPUT_NODES; i = i + 1) begin
			weights[(OUTPUT_NODES-1-i)*DATA_WIDTH+:DATA_WIDTH] = memory[(address*OUTPUT_NODES)+i];
		end
	end
end

initial begin
	$readmemh(file,memory);
end

endmodule
