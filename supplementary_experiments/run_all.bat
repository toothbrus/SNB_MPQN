@echo off
echo =============================================
echo   Distributed Quantum Circuit Simulator
echo   Batch Experiment Runner
echo =============================================
echo.

:: 设置重复运行次数和输出文件前缀
set RUNS=1
set PREFIX=myresults

echo [*] RUNS = %RUNS%
echo [*] PREFIX = %PREFIX%
echo.

echo Running exp1 (default comparison) ...
python new_simulation.py --experiment exp1 --runs %RUNS% --out-prefix %PREFIX%_exp1


echo Scanning t_init ...
python new_simulation.py --experiment exp_parameter_sweep --param-name t_init --param-values "10,100,250,500" --runs %RUNS% --out-prefix %PREFIX%_tinit

echo Scanning p_c ...
python new_simulation.py --experiment exp_parameter_sweep --param-name p_c --param-values "0.08,0.12,0.16,0.20" --runs %RUNS% --out-prefix %PREFIX%_pc

echo Scanning p_t ...
python new_simulation.py --experiment exp_parameter_sweep --param-name p_t --param-values "0.7,0.8,0.9,0.99" --run %RUNS% --out-prefix %PREFIX%_p_t

echo Scanning t_deco ...
python new_simulation.py --experiment exp_parameter_sweep --param-name t_deco --param-values "500,1000,2100,4200" --runs %RUNS% --out-prefix %PREFIX%_tdeco

echo Scanning t_retry ...
python new_simulation.py --experiment exp_parameter_sweep --param-name t_retry --param-values "0.1,0.5,1.0,5.0" --runs %RUNS% --out-prefix %PREFIX%_retry





echo =============================================
echo   All experiments finished!
echo =============================================
pause