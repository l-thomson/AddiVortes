// Routine registration is forwarded from C to the Rust static library so
// the linker keeps it.

void R_init_thiessen_extendr(void *dll);
void register_extendr_panic_hook(void);

void R_init_thiessen(void *dll) {
    register_extendr_panic_hook();
    R_init_thiessen_extendr(dll);
}
