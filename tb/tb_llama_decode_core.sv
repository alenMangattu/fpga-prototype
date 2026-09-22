`timescale 1ns/1ps

module tb_llama_decode_core;
    localparam integer D=4, FF=4, H=2, KH=1, L=2, V=4, C=2;
    localparam integer FRAC=16;
    reg clk=0, reset_n=0, clear_cache=0, start=0, cfg_we=0;
    reg [$clog2(V)-1:0] input_token=0;
    reg [30:0] cfg_addr=0;
    reg [31:0] cfg_wdata=0;
    wire [31:0] cfg_rdata;
    wire cfg_ready, busy, done;
    wire [$clog2(V)-1:0] output_token;
    wire [$clog2(C+1)-1:0] position;
    wire [31:0] cycle_count, parameter_words;
    integer n, l, base, layer_base, layer_words, final_norm;
    integer q_base, k_base, v_base, o_base, ffn_norm_base;
    integer gate_base, up_base, down_base;

    llama_decode_core #(
        .D_MODEL(D), .D_FF(FF), .HEADS(H), .KV_HEADS(KH),
        .LAYERS(L), .VOCAB_SIZE(V), .MAX_CONTEXT(C), .FRAC(FRAC)
    ) dut (
        .clk(clk), .reset_n(reset_n), .clear_cache(clear_cache),
        .start(start), .input_token(input_token),
        .cfg_we(cfg_we), .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata),
        .cfg_rdata(cfg_rdata), .cfg_ready(cfg_ready), .busy(busy),
        .done(done), .output_token(output_token), .position(position),
        .cycle_count(cycle_count), .parameter_words(parameter_words)
    );

    always #5 clk = ~clk;

    task write_word(input integer address, input signed [31:0] value);
        begin
            @(negedge clk);
            cfg_addr = address;
            cfg_wdata = value;
            cfg_we = 1;
            @(negedge clk);
            cfg_we = 0;
        end
    endtask

    task run_token(input integer token, input integer expected, input integer expected_position);
        begin
            @(negedge clk);
            input_token = token;
            start = 1;
            @(negedge clk);
            start = 0;
            wait(done);
            if (output_token !== expected[$clog2(V)-1:0]) begin
                $display("FAIL token=%0d expected=%0d got=%0d",token,expected,output_token);
                $fatal(1);
            end
            if (position !== expected_position[$clog2(C+1)-1:0]) begin
                $display("FAIL position expected=%0d got=%0d",expected_position,position);
                $fatal(1);
            end
            $display("token %0d -> %0d in %0d RTL cycles",token,output_token,cycle_count);
        end
    endtask

    initial begin
        repeat(3) @(negedge clk);
        reset_n = 1;

        // Zero every parameter, then construct a deterministic one-hot model.
        for (n=0; n<parameter_words; n=n+1)
            write_word(n,0);
        write_word(0,1);                 // RMS epsilon, one Q16.16 LSB
        write_word(1,32'sd46341);        // 1/sqrt(2)

        // Tied one-hot embeddings at global base 2.
        for (n=0; n<V; n=n+1)
            write_word(2+n*D+n,1 <<< FRAC);

        // RoPE cos=1, sin=0 for both supported positions.
        base = 2 + V*D;
        for (n=0; n<C*(D/H/2); n=n+1)
            write_word(base+n,1 <<< FRAC);

        // Per-layer norm vectors are one.
        base = 2 + V*D + 2*C*(D/H/2);
        layer_words = D + D*D + (KH*(D/H))*D*2 + D*D + D + FF*D*2 + D*FF;
        for (l=0; l<L; l=l+1) begin
            layer_base = base + l*layer_words;
            q_base = layer_base + D;
            k_base = q_base + D*D;
            v_base = k_base + (KH*(D/H))*D;
            o_base = v_base + (KH*(D/H))*D;
            ffn_norm_base = o_base + D*D;
            gate_base = ffn_norm_base + D;
            up_base = gate_base + FF*D;
            down_base = up_base + FF*D;
            for (n=0; n<D; n=n+1) begin
                write_word(layer_base+n,1 <<< FRAC);
                write_word(ffn_norm_base+n,1 <<< FRAC);
                // Q/O and all three SwiGLU matrices are identities. K/V select
                // the first KV_DIM channels. This makes attention and FFN math
                // nonzero while retaining an obvious winning output token.
                write_word(q_base+n*D+n,1 <<< FRAC);
                write_word(o_base+n*D+n,1 <<< FRAC);
                write_word(gate_base+n*D+n,1 <<< FRAC);
                write_word(up_base+n*D+n,1 <<< FRAC);
                write_word(down_base+n*FF+n,1 <<< FRAC);
            end
            for (n=0; n<KH*(D/H); n=n+1) begin
                write_word(k_base+n*D+n,1 <<< FRAC);
                write_word(v_base+n*D+n,1 <<< FRAC);
            end
        end
        final_norm = base + L*layer_words;
        for (n=0; n<D; n=n+1)
            write_word(final_norm+n,1 <<< FRAC);

        run_token(1,1,1);
        if ($signed(dut.attention[1]) <= 0 || $signed(dut.ff[1]) <= 0)
            $fatal(1,"nonzero attention/SwiGLU datapath was not exercised");
        run_token(3,3,2);
        if ($signed(dut.probabilities[0]) <= 0 ||
            $signed(dut.probabilities[1]) <= 0)
            $fatal(1,"multi-token softmax probabilities were not produced");

        @(negedge clk);
        clear_cache=1;
        @(negedge clk);
        clear_cache=0;
        if (position !== 0) $fatal(1,"cache clear did not reset position");

        $display("PASS complete RTL Llama graph; parameter words=%0d",parameter_words);
        $finish;
    end

    initial begin
        #2000000;
        $fatal(1,"timeout");
    end
endmodule
