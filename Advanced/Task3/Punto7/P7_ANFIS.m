%% PUNTO 7 — Sistema neurodifuso ANFIS para el riesgo de estrés hídrico del maíz (Tibaitatá)
% Hidroinformática Avanzada · Tarea 3 · Autor: Douglas A. Gómez-Latorre
% Tabla de patrones: la misma de los puntos 5 y 6 (tabla_patrones_Zr1.0.csv).
% Requisitos: Fuzzy Logic Toolbox (genfis, genfisOptions, anfis, anfisOptions, evalfis,
% gensurf, plotmf, showrule). R2019b o posterior.
% ESTADO: código propuesto. Las cifras del informe provienen de una réplica en Python del
% aprendizaje híbrido de anfis (LSE + gradiente con paso adaptativo). La inicialización de
% genfis y la regla de paso de MATLAB pueden diferir ligeramente: confirmar con este script.

clear; clc; close all; rng(2026,'twister');

%% 1. Datos (sin normalizar: anfis trabaja en las unidades originales)
T  = readtable('tabla_patrones_Zr1.0.csv');
IN = {'fD_t','ET0_t1_mm','P_t1_mm'};
C = T(strcmp(T.conjunto,'calibracion'),:);  V = T(strcmp(T.conjunto,'validacion'),:);
P = T(strcmp(T.conjunto,'prueba'),:);
Xc = C{:,IN}; dc = C.R_t1;  Xv = V{:,IN}; dv = V.R_t1;  Xp = P{:,IN}; dp = P.R_t1;

%% 2. Búsqueda de estructura (decidida SOLO con validación)
% Etapa 1: el mismo número de funciones por entrada (2, 3 o 4), gaussmf o gbellmf,
%          consecuente constante o lineal, 100 épocas.
% Etapa 2: particiones distintas por entrada (f_D domina), consecuente lineal, 300 épocas.
cfg = {[2 2 2],[3 2 2],[4 2 2],[5 2 2],[3 3 2],[3 2 3]};
tipos = {'gaussmf','gbellmf'};
mejor = struct('rmse',inf);
for a = 1:numel(cfg)
  for b = 1:numel(tipos)
    gopt = genfisOptions('GridPartition');
    gopt.NumMembershipFunctions      = cfg{a};
    gopt.InputMembershipFunctionType = tipos{b};
    gopt.OutputMembershipFunctionType = 'linear';      % Sugeno de primer orden
    fis0 = genfis(Xc, dc, gopt);
    aopt = anfisOptions('InitialFIS', fis0, 'EpochNumber', 300, ...
        'ValidationData', [Xv dv], 'OptimizationMethod', 1, ...   % 1 = híbrido (LSE + gradiente)
        'InitialStepSize', 0.01, 'StepSizeIncreaseRate', 1.1, 'StepSizeDecreaseRate', 0.9, ...
        'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0);
    [~, errC, ~, fisV, errV] = anfis([Xc dc], aopt);   % fisV = FIS de la época con menor error de validación
    [mv, kv] = min(errV);
    fprintf('MF %s %-7s: reglas %2d | RMSE cal %.4f | RMSE val %.4f (época %d)\n', ...
        mat2str(cfg{a}), tipos{b}, prod(cfg{a}), errC(kv), mv, kv);
    if mv < mejor.rmse
        mejor = struct('rmse',mv,'fis',fisV,'fis0',fis0,'cfg',cfg{a},'tipo',tipos{b}, ...
                       'errC',errC,'errV',errV,'k',kv);
    end
  end
end
% Réplica Python: mejor = [4 2 2] gaussmf, 16 reglas, RMSE val 0,0051 en la época 138.

%% 3. Main program: modelo elegido y evaluación final
fis = mejor.fis;
fprintf('\nElegido: %s %s (%d reglas)\n', mat2str(mejor.cfg), mejor.tipo, prod(mejor.cfg));
figure('Name','Curvas'); semilogy(mejor.errC,'LineWidth',1.5); hold on; semilogy(mejor.errV,'LineWidth',1.5);
xline(mejor.k,'--',sprintf('época %d',mejor.k)); xlabel('Época'); ylabel('RMSE');
legend('Calibración','Validación'); title('Aprendizaje híbrido de ANFIS');

conj = {'Calibración',Xc,dc; 'Validación',Xv,dv; 'Prueba',Xp,dp};
figure('Name','Observado vs predicho');
for j = 1:3
    o = conj{j,3}; yh = evalfis(fis, conj{j,2}); e = yh - o;
    m = [sqrt(mean(e.^2)) mean(abs(e)) mean(e) 1-sum(e.^2)/sum((o-mean(o)).^2)];
    fprintf('%-12s RMSE %.4f | MAE %.4f | sesgo %+.4f | R2 %.4f\n', conj{j,1}, m);
    subplot(1,3,j); scatter(o,yh,6,'filled','MarkerFaceAlpha',0.3); hold on; plot([0 1],[0 1],'k--');
    axis([-0.05 1 -0.05 1]); axis square; xlabel('R observado'); ylabel('R predicho (ANFIS)');
    title(sprintf('%s: RMSE %.4f', conj{j,1}, m(1)));
end
% Réplica Python (prueba): RMSE 0,0053 | MAE 0,0027 | sesgo -0,0005 | R2 0,9997

%% 4. Funciones de pertenencia iniciales y entrenadas; reglas
figure('Name','Funciones de pertenencia');
for i = 1:3
    subplot(2,3,i);   plotmf(mejor.fis0,'input',i); title(['Inicial: ' IN{i}],'Interpreter','none');
    subplot(2,3,i+3); plotmf(fis,'input',i);        title(['Entrenada: ' IN{i}],'Interpreter','none');
end
showrule(fis)

%% 5. Superficies de respuesta
figure('Name','Superficies');
subplot(1,2,1); gensurf(fis, gensurfOptions('InputIndex',[1 2],'ReferenceInputs',[NaN NaN 0]));
title('ANFIS: f_D x ET_0, sin lluvia');
subplot(1,2,2); gensurf(fis, gensurfOptions('InputIndex',[1 3],'ReferenceInputs',[NaN 2.7 NaN]));
title('ANFIS: f_D x lluvia, ET_0 = 2,7');

%% 6. Ejemplo de sobreparametrización: 4 x 4 x 4 = 64 reglas lineales (280 parámetros)
gopt = genfisOptions('GridPartition'); gopt.NumMembershipFunctions = [4 4 4];
gopt.InputMembershipFunctionType = 'gaussmf'; gopt.OutputMembershipFunctionType = 'linear';
aopt = anfisOptions('InitialFIS', genfis(Xc,dc,gopt), 'EpochNumber', 100, ...
    'ValidationData', [Xv dv], 'DisplayANFISInformation',0,'DisplayErrorValues',0, ...
    'DisplayStepSize',0,'DisplayFinalResults',0);
[~, eC64, ~, ~, eV64] = anfis([Xc dc], aopt);
figure('Name','Sobreparametrización'); semilogy(eC64); hold on; semilogy(eV64);
legend('Calibración','Validación'); xlabel('Época'); ylabel('RMSE'); title('ANFIS con 64 reglas');
% Réplica Python: RMSE de calibración 0,003 frente a RMSE de validación de 0,8 a 50

%% 7. Comparación final en la MISMA prueba (salida acotada a [0,1] en todos los modelos)
% Requiere los resultados de los puntos 4 a 6 en el espacio de trabajo o guardados en disco.
