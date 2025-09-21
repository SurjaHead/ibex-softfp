`uv venv my-env`

`source my-env/bin/activate`

`uv pip install -r python-requirements.txt`

TO BUILD THE RTL:

```
export CPPFLAGS="-I/opt/homebrew/include" && export LDFLAGS="-L/opt/homebrew/lib -lelf" && fusesoc --cores-root=. run --target=sim --setup --build lowrisc:ibex:ibex_simple_system $(util/ibex_config.py small fusesoc_opts)
```

TO MAKE THE C FILE:

```
make -C examples/sw/simple_system/relu_test
```

TO RUN THE SIMULATION:

```
./build/lowrisc_ibex_ibex_simple_system_0/sim-verilator/Vibex_simple_system -t --meminit=ram,examples/sw/simple_system/relu_test/relu_test.vmem
```
