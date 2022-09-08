// Ȩ�ض�ȡ

module weightMemory(clk,address,weights);

// ע���ⶼ��Ĭ�ϲ����������ʱ����Ըı�
parameter DATA_WIDTH = 32; // ����λ��
parameter INPUT_NODES = 100; // ����ڵ���Ŀ
parameter OUTPUT_NODES = 32; // ����ڵ���Ŀ
parameter file = "E:/Parameters With No Seperators/weights1_IEEE.txt"; //����Ȩ���ĵ�

localparam TOTAL_WEIGHT_SIZE = INPUT_NODES * OUTPUT_NODES; // �ܹ���x��Ȩ��

input clk; //ʱ���ź�
input [8:0] address; // λ��Ϊ8�����Ϊ1
output reg [DATA_WIDTH*OUTPUT_NODES-1:0] weights; // �洢Ȩ�أ�λ��Ϊx�����Ϊ1

reg [DATA_WIDTH-1:0] memory [0:TOTAL_WEIGHT_SIZE-1]; // 32λλ���洢Ȩ��

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
