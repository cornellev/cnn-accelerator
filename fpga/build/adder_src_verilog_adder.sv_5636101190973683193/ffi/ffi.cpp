
#include <cstring> // std::memcpy
#include "verilated.h"
#include "Vadder.h"

extern "C" {
    void*  ffi_new_Vadder() {
        return new Vadder{};
    }

    
    void ffi_Vadder_eval(Vadder* top) {
        top->eval();
    }

    void ffi_delete_Vadder(Vadder* top) {
        delete top;
    }


    void ffi_Vadder_pin_clk(Vadder* top, VL_IN8(new_value, 1, 0)) {
        top->clk = new_value;
    }
            

    void ffi_Vadder_pin_a(Vadder* top, VL_IN8(new_value, 3, 0)) {
        top->a = new_value;
    }
            

    void ffi_Vadder_pin_b(Vadder* top, VL_IN8(new_value, 3, 0)) {
        top->b = new_value;
    }
            

    VL_OUT8(/* return value */, 3, 0) ffi_Vadder_read_c(Vadder* top) {
        return top->c;
    }
            
} // extern "C"
