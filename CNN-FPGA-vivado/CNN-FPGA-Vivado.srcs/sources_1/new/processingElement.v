module processingElement(clk,reset,floatA,floatB,result);
// PEģ�飬������Ĵ���Ԫ

parameter DATA_WIDTH = 32;

input clk, reset;
input [DATA_WIDTH-1:0] floatA, floatB;
output reg [DATA_WIDTH-1:0] result; // �Ĵ�������

wire [DATA_WIDTH-1:0] multResult;
wire [DATA_WIDTH-1:0] addResult;

// ʱ���߼���·��ѭ���ۼ�
floatMult FM (floatA,floatB,multResult);
floatAdd FADD (multResult,result,addResult);

always @ (posedge clk or posedge reset) begin
	if (reset == 1'b1) begin
		result = 0; // ��λ
	end else begin
		result = addResult; // ��������н��и�ֵ
	end
end

endmodule
