// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vadder.h for the primary calling header

#include "Vadder__pch.h"

void Vadder___024root___ctor_var_reset(Vadder___024root* vlSelf);

Vadder___024root::Vadder___024root(Vadder__Syms* symsp, const char* namep)
 {
    vlSymsp = symsp;
    vlNamep = strdup(namep);
    // Reset structure values
    Vadder___024root___ctor_var_reset(this);
}

void Vadder___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vadder___024root::~Vadder___024root() {
    VL_DO_DANGLING(std::free(const_cast<char*>(vlNamep)), vlNamep);
}
