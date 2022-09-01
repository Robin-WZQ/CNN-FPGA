module layer(clk,reset,input_fc,weights,output_fc); // 第一层全连接

parameter DATA_WIDTH = 32; // 数据位宽
parameter INPUT_NODES = 100; // 输入节点：100个
parameter OUTPUT_NODES = 32; // 输出节点： 32个

input clk, reset;
input [DATA_WIDTH*INPUT_NODES-1:0] input_fc; // 位宽为3200，深度为1
input [DATA_WIDTH*OUTPUT_NODES-1:0] weights; // 同上
output [DATA_WIDTH*OUTPUT_NODES-1:0] output_fc; // 同上

reg [DATA_WIDTH-1:0] selectedInput;
integer j;
// 生成语句，重复操作，如此优雅！
genvar i;

generate
	for (i = 0; i < OUTPUT_NODES; i = i + 1) begin
		processingElement PE 
		(
			.clk(clk),
			.reset(reset),
			.floatA(selectedInput),
			.floatB(weights[DATA_WIDTH*i+:DATA_WIDTH]),
			.result(output_fc[DATA_WIDTH*i+:DATA_WIDTH])
		);
	end
endgenerate

always @ (posedge clk or posedge reset) begin
	if (reset == 1'b1) begin
		selectedInput = 0;
		j = INPUT_NODES - 1;
	end else if (j < 0) begin
		selectedInput = 0;
	end else begin
		selectedInput = input_fc[DATA_WIDTH*j+:DATA_WIDTH];
		j = j - 1;
	end
end

endmodule
