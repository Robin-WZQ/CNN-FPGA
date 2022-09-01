module IEEE162IEEE32(clk,reset,input_fc,output_fc);

parameter DATA_WIDTH_1 = 16;
parameter DATA_WIDTH_2 = 32;
parameter NODES = 400;

input clk, reset;
input [DATA_WIDTH_1*NODES-1:0] input_fc;
output reg [DATA_WIDTH_2*NODES-1:0] output_fc;

reg [7:0] temp;
integer i;

always @ (negedge clk or posedge reset) begin
	if (reset == 1'b1) begin
		output_fc = 0;
	end else begin
        temp = 8'b00000000;
        for (i = 0; i < NODES; i = i + 1) begin
                output_fc[DATA_WIDTH_2*(i+1)-1] = input_fc[DATA_WIDTH_1*(i+1)-1];
                temp[0+:5] = input_fc[(DATA_WIDTH_1*i+10)+:5];
                output_fc[(DATA_WIDTH_2*i+23)+:8] = temp + 8'b01110000;
                output_fc[(DATA_WIDTH_2*i+13)+:10] = input_fc[(DATA_WIDTH_1*i+0)+:10];
                output_fc[(DATA_WIDTH_2*i)+:13] = 0;
        end

	end
end

endmodule
