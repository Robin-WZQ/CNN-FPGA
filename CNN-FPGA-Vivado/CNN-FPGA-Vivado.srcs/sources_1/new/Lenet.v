module Lenet(clk,reset,CNNinput,Conv1F,Conv2F,LeNetoutput);

parameter DATA_WIDTH_1 = 16;
parameter DATA_WIDTH_2 = 32;
parameter ImgInW = 32;
parameter ImgInH = 32;
parameter Kernel = 5;
parameter MP2out = 5;
parameter DepthC1 = 6;
parameter DepthC2 = 16;

integer counter;

input clk, reset;
input [ImgInW*ImgInH*DATA_WIDTH_1-1:0] CNNinput; // lenet������
input [Kernel*Kernel*DepthC1*DATA_WIDTH_1-1:0] Conv1F; // ��һ������Ȩ��
input [DepthC2*Kernel*Kernel*DepthC1*DATA_WIDTH_1-1:0] Conv2F; // �ڶ�������Ȩ��
output [3:0] LeNetoutput; // �����10����

reg reset1,reset2; // �����ȫ���ӵĸ�λ�ź�

wire [MP2out*MP2out*DepthC2*DATA_WIDTH_1-1:0] CNNout;  // �������������5*5*16*16=6400
wire [MP2out*MP2out*DepthC2*DATA_WIDTH_2-1:0] ANNin;  // �������������5*5*16*16=6400

integrationConv C1
(
    .clk(clk),
    .reset(reset1),
    .CNNinput(CNNinput),
    .Conv1F(Conv1F),
    .Conv2F(Conv2F),
    .iConvOutput(CNNout)
);

IEEE162IEEE32
#(.NODES(400))
  T1
  (
    .clk(clk),
    .reset(reset),
    .input_fc(CNNout),
    .output_fc(ANNin)
  );
  
ANNfull A1
(
    .clk(clk),
    .reset(reset2),
    .input_ANN(ANNin),
    .output_ANN(LeNetoutput)
);

always @(posedge clk or posedge reset) begin
  if (reset == 1'b1) begin
     reset1 = 1'b1;
     reset2 = 1'b1;
     counter = 0;
  end
else begin
  counter = counter + 1;
  if (counter < 7*1457+6*784*6+8+18*22*152 + 6*1600 + 20 + 10000) begin
    reset1 = 1'b0;
    end
   else begin
    reset2 = 1'b0;
    end
   end
 end
endmodule
