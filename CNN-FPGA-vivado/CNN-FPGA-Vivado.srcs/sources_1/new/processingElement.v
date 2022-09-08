module processingElement(clk,reset,floatA,floatB,result);
// PE模块，最基础的处理单元

parameter DATA_WIDTH = 32;

input clk, reset;
input [DATA_WIDTH-1:0] floatA, floatB;
output reg [DATA_WIDTH-1:0] result; // 寄存器类型

wire [DATA_WIDTH-1:0] multResult;
wire [DATA_WIDTH-1:0] addResult;

// 时序逻辑电路，循环累加
floatMult FM (floatA,floatB,multResult);
floatAdd FADD (multResult,result,addResult);

always @ (posedge clk or posedge reset) begin
	if (reset == 1'b1) begin
		result = 0; // 复位
	end else begin
		result = addResult; // 过程语句中进行赋值
	end
end

endmodule
