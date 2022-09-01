module max (n1,n2,n3,n4,max);

parameter DATA_WIDTH = 16;

input [DATA_WIDTH-1:0]n1;
input [DATA_WIDTH-1:0]n2;
input [DATA_WIDTH-1:0]n3;
input [DATA_WIDTH-1:0]n4;
output [DATA_WIDTH-1:0]max;
wire [DATA_WIDTH-1:0]max12;
wire [DATA_WIDTH-1:0]max34;

assign max12=(n1>n2)?n1:n2;
assign max34=(n3>n4)?n3:n4;
assign max = (max12>=max34) ? max12 : max34;
endmodule