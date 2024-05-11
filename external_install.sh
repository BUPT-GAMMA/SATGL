cd ./satgl/external/abc
make
cd ../open-wbo
make 
cd ../aiger/aiger/
make
cd ../cnf2aig/
make 
cd ../../picosat-965
make

# sls solver
cd ../sls_solver/probSAT
make