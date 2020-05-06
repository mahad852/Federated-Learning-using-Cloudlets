bash run_fedavg.sh mnist 0 0.03 mclr 100 | tee mnist/fedavg_drop0
bash run_fedprox.sh mnist 0 0 0.03 mclr 100 | tee mnist/fedprox_drop0_mu0
bash run_fedprox.sh mnist 0 1 0.03 mclr 100 | tee mnist/fedprox_drop0_mu1

bash run_fedavg.sh mnist 0.5 0.03 mclr 100 | tee mnist/fedavg_drop0.5
bash run_fedprox.sh mnist 0.5 0 0.03 mclr 100 | tee mnist/fedprox_drop0.5_mu0
bash run_fedprox.sh mnist 0.5 1 0.03 mclr 100 | tee mnist/fedprox_drop0.5_mu1

bash run_fedavg.sh mnist 0.9 0.03 mclr 100 | tee mnist/fedavg_drop0.9
bash run_fedprox.sh mnist 0.9 0 0.03 mclr 100 | tee mnist/fedprox_drop0.9_mu0
bash run_fedprox.sh mnist 0.9 1 0.03 mclr 100 | tee mnist/fedprox_drop0.9_mu1
