module FindMax(n,max);

parameter DATA_WIDTH = 320;

input [DATA_WIDTH-1:0]n;
output [3:0]max;

wire [35:0] n1;
wire [35:0] n2;
wire [35:0] n3;
wire [35:0] n4;
wire [35:0] n5;
wire [35:0] n6;
wire [35:0] n7;
wire [35:0] n8;
wire [35:0] n9;
wire [35:0] n10;

assign n1[35:32] = 4'b0000;
assign n2[35:32] = 4'b0001;
assign n3[35:32] = 4'b0010;
assign n4[35:32] = 4'b0011;
assign n5[35:32] = 4'b0100;
assign n6[35:32] = 4'b0101;
assign n7[35:32] = 4'b0110;
assign n8[35:32] = 4'b0111;
assign n9[35:32] = 4'b1000;
assign n10[35:32] = 4'b1001;

assign n1[31:0] = n[31:0];
assign n2[31:0] = n[63:32];
assign n3[31:0] = n[95:64];
assign n4[31:0] = n[127:96];
assign n5[31:0] = n[159:128];
assign n6[31:0] = n[191:160];
assign n7[31:0] = n[223:192];
assign n8[31:0] = n[255:224];
assign n9[31:0] = n[287:256];
assign n10[31:0] = n[319:288];

wire [35:0]max12;
wire [35:0]max34;
wire [35:0]max56;
wire [35:0]max78;
wire [35:0]max90;
//第二层
wire [35:0]max14;
wire [35:0]max58;
//第三层
wire [35:0]max18;
//第四层
wire [35:0]max10;


assign max12=(n1[31:0]>n2[31:0])?n1:n2;
assign max34=(n3[31:0]>n4[31:0])?n3:n4;
assign max56=(n5[31:0]>n6[31:0])?n5:n6;
assign max78=(n7[31:0]>n8[31:0])?n7:n8;
assign max90=(n9[31:0]>n10[31:0])?n9:n10;

assign max14=(max12[31:0]>max34[31:0])?max12:max34;
assign max58=(max56[31:0]>max78[31:0])?max56:max78;

assign max18=(max14[31:0]>max58[31:0])?max14:max58;

assign max10=(max18[31:0]>max90[31:0])?max18:max90;

assign max = max10[35:32];
endmodule