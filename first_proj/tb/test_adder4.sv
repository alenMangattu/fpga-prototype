module test_adder4;
    logic [3:0] a, b;
    logic [4:0] sum;

    adder4 dut (.a(a), .b(b), .sum(sum));

    initial begin
        for (int x = 0; x < 16; x++) begin
            for (int y = 0; y < 16; y++) begin
                a = x;
                b = y;
                #1;
                if (sum !== x + y)
                    $fatal(1, "%0d + %0d: got %0d", x, y, sum);
            end
        end
        $display("PASS: all 256 input pairs");
        $finish;
    end
endmodule
