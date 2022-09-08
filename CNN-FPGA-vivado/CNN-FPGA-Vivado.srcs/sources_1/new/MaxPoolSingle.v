`timescale 1 ns / 10 ps

module MaxPoolSingle(aPoolIn,aPoolOut);
  
parameter DATA_WIDTH = 16;
parameter InputH = 28;
parameter InputW = 28;
parameter Depth = 1;

input [0:InputH*InputW*Depth*DATA_WIDTH-1] aPoolIn;
output [0:(InputH/2)*(InputW/2)*Depth*DATA_WIDTH-1] aPoolOut;

genvar i,j;

generate 
  for (i=0; i<(InputH); i=i+2) begin
    for (j=0; j<(InputW); j=j+2) begin
    max
    #(
     .DATA_WIDTH(DATA_WIDTH)
     ) max1
    (
      .n1(aPoolIn[(i*InputH+j)*DATA_WIDTH+:DATA_WIDTH]),
      .n2(aPoolIn[(i*InputH+j+1)*DATA_WIDTH+:DATA_WIDTH]),
      .n3(aPoolIn[((i+1)*InputH+j)*DATA_WIDTH+:DATA_WIDTH]),
      .n4(aPoolIn[((i+1)*InputH+j+1)*DATA_WIDTH+:DATA_WIDTH]),
      .max(aPoolOut[(i/2*InputH/2+j/2)*DATA_WIDTH+:DATA_WIDTH])
      );
    end
  end
endgenerate

endmodule
      