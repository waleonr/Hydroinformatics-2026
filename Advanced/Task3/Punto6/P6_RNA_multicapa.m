%% PUNTO 6 — Red neuronal artificial CON capa oculta (perceptrón multicapa 3-H-1)
% Hidroinformática Avanzada · Tarea 3 · Autor: Douglas A. Gómez-Latorre
% Problema y tabla de patrones: los mismos del punto 5 (tabla_patrones_Zr1.0.csv).
% Algoritmo: retropropagación con la regla delta generalizada, actualización por patrón,
% sigmoide en la capa oculta y en la salida, con momento.
%
% Sección 3: MAIN PROGRAM con el algoritmo programado explícitamente (sin toolbox).
% Sección 7: equivalente con Deep Learning Toolbox (feedforwardnet / adapt / train).
% ESTADO: código propuesto. Las cifras del informe provienen de una réplica en Python
% (compilada con numba) del mismo algoritmo.

clear; clc; close all;

%% 1. Datos, partición cronológica y normalización (idénticas al punto 5)
T  = readtable('tabla_patrones_Zr1.0.csv');
IN = {'fD_t','ET0_t1_mm','P_t1_mm'};
C = T(strcmp(T.conjunto,'calibracion'),:);  V = T(strcmp(T.conjunto,'validacion'),:);
P = T(strcmp(T.conjunto,'prueba'),:);
lo = min(C{:,IN}); hi = max(C{:,IN}); dlo = min(C.R_t1); dhi = max(C.R_t1);
nx = @(X) (X-lo)./(hi-lo);  nd = @(d) 0.1+0.8*(d-dlo)/(dhi-dlo);  invd = @(y) dlo+(y-0.1)/0.8*(dhi-dlo);
Xc = nx(C{:,IN}); Xv = nx(V{:,IN}); Xp = nx(P{:,IN});  dc = nd(C.R_t1); dv = nd(V.R_t1);

%% 2. Optimización de hiperparámetros (SOLO con validación)
% Etapa 1: H en {1,2,3,4,6,8,12}, η en {0.05,0.1,0.3}, α en {0,0.8}; 400 épocas; 3 semillas.
% Etapa 2: H en {2,3,4,6,8},      η en {0.3,0.5},     α en {0.8,0.9}; 2.000 épocas; 3 semillas.
% Criterio: menor MSE de validación medio entre semillas (época elegida por parada temprana).
% (Para no repetir horas de cómputo, aquí se ejecuta solo la etapa 2; la etapa 1 es idéntica
%  cambiando las listas.)
grid = combvec([2 3 4 6 8],[0.3 0.5],[0.8 0.9])';
res  = zeros(size(grid,1),2);
for g = 1:size(grid,1)
    v = zeros(3,1);
    for s = 1:3
        [~,hv] = entrenarMLP(Xc,dc,Xv,dv,grid(g,1),grid(g,2),grid(g,3),2000,s);
        v(s) = min(hv);
    end
    res(g,:) = [mean(v) std(v)];
    fprintf('H=%2d η=%.1f α=%.1f -> MSE_val = %.2e (DE %.1e)\n', grid(g,:), res(g,:));
end
[~,ib] = min(res(:,1));  H = grid(ib,1); eta = grid(ib,2); alfa = grid(ib,3);
% Réplica Python: H = 6, η = 0,5, α = 0,9 (MSE_val = 1,3e-5)

%% 3. MAIN PROGRAM: entrenamiento final con parada temprana
nEp = 5000;
[hc,hv,W1,W2,k] = entrenarMLP(Xc,dc,Xv,dv,H,eta,alfa,nEp,0);
fprintf('Arquitectura 3-%d-1, η=%.1f, α=%.1f, época elegida %d\n', H, eta, alfa, k);
figure('Name','Curvas'); loglog(hc,'LineWidth',1.2); hold on; loglog(hv,'LineWidth',1.2);
xlabel('Época'); ylabel('MSE (normalizado)'); legend('Calibración','Validación');
title(sprintf('Perceptrón 3-%d-1', H));

%% 4. Evaluación en calibración, validación y prueba (escala original de R)
sig  = @(z) 1./(1+exp(-z));
pred = @(X) invd(sig([ones(size(X,1),1) sig([ones(size(X,1),1) X]*W1')]*W2));
conj = {'Calibración',C,Xc; 'Validación',V,Xv; 'Prueba',P,Xp};
figure('Name','Observado vs predicho');
for j = 1:3
    o = conj{j,2}.R_t1; yh = pred(conj{j,3}); e = yh - o;
    m = [sqrt(mean(e.^2)) mean(abs(e)) mean(e) 1-sum(e.^2)/sum((o-mean(o)).^2)];
    fprintf('%-12s RMSE %.4f | MAE %.4f | sesgo %+.4f | R2 %.4f\n', conj{j,1}, m);
    subplot(1,3,j); scatter(o,yh,6,'filled','MarkerFaceAlpha',0.3); hold on; plot([0 1],[0 1],'k--');
    axis([0 1 0 1]); axis square; xlabel('R observado'); ylabel('R predicho');
    title(sprintf('%s: RMSE %.4f', conj{j,1}, m(1)));
end
% Réplica Python (prueba): RMSE 0,0029 | MAE 0,0018 | sesgo -0,0001 | R2 0,9999

%% 5. Comparación con la red sin capa oculta (punto 5) en la MISMA prueba
% Ejecutar antes P5_RNA_sin_capa_oculta.m y guardar w5 (pesos de la época elegida):
%   save('pesos_P5.mat','w5')
if isfile('pesos_P5.mat')
    load('pesos_P5.mat','w5');
    y5 = invd(sig([ones(size(Xp,1),1) Xp]*w5));
    fprintf('Prueba: RMSE sin capa oculta %.4f | multicapa %.4f\n', ...
        sqrt(mean((y5-P.R_t1).^2)), sqrt(mean((pred(Xp)-P.R_t1).^2)));
end
% Réplica Python: 0,0373 frente a 0,0029

%% 6. Superficie de respuesta del perceptrón (sin lluvia)
[F,E] = meshgrid(linspace(0,1,41), linspace(1.5,4.8,31));
Z = reshape(pred(nx([F(:) E(:) zeros(numel(F),1)])), size(F));
figure('Name','Superficie MLP'); surf(F,E,Z); xlabel('f_D'); ylabel('ET_0 t+1'); zlabel('Riesgo');
title('Perceptrón multicapa, P = 0');

%% 7. Equivalente con Deep Learning Toolbox
% (a) Aprendizaje por patrón con momento, como en clase: adapt + learngdm
net = feedforwardnet(H);
net.layers{1}.transferFcn = 'logsig';     % sigmoide en la capa oculta
net.layers{2}.transferFcn = 'logsig';     % sigmoide en la salida (por defecto sería purelin)
net.inputs{1}.processFcns  = {};          % ya normalizamos manualmente
net.outputs{2}.processFcns = {};
net.adaptFcn = 'adaptwb';
for L = 1:2, net.inputWeights{1,1}.learnFcn = 'learngdm'; net.layerWeights{2,1}.learnFcn = 'learngdm';
    net.biases{L}.learnFcn = 'learngdm'; end
net.inputWeights{1,1}.learnParam.lr = eta; net.inputWeights{1,1}.learnParam.mc = alfa;
net.layerWeights{2,1}.learnParam.lr = eta; net.layerWeights{2,1}.learnParam.mc = alfa;
net.biases{1}.learnParam.lr = eta; net.biases{1}.learnParam.mc = alfa;
net.biases{2}.learnParam.lr = eta; net.biases{2}.learnParam.mc = alfa;
net = configure(net, Xc', dc');
Xseq = con2seq(Xc'); Dseq = con2seq(dc');   % secuencia => una actualización por patrón
for ep = 1:200                               % (pocas épocas: adapt es lento)
    idx = randperm(numel(Xseq));
    net = adapt(net, Xseq(idx), Dseq(idx));
end
fprintf('Toolbox (adapt, 200 épocas): RMSE prueba %.4f\n', sqrt(mean((invd(net(Xp')') - P.R_t1).^2)));

% (b) Alternativa por lotes, más rápida: train con traingdx o Levenberg-Marquardt,
%     y parada temprana con la partición propia (divideind)
Xall = [Xc; Xv]'; Dall = [dc; dv]';
net2 = feedforwardnet(H,'trainlm');
net2.layers{2}.transferFcn = 'logsig';
net2.inputs{1}.processFcns = {}; net2.outputs{2}.processFcns = {};
net2.divideFcn = 'divideind';
net2.divideParam.trainInd = 1:size(Xc,1);
net2.divideParam.valInd   = size(Xc,1) + (1:size(Xv,1));
net2.divideParam.testInd  = [];
net2.trainParam.epochs = 1000; net2.trainParam.max_fail = 50;
net2 = train(net2, Xall, Dall);
fprintf('Toolbox (trainlm): RMSE prueba %.4f\n', sqrt(mean((invd(net2(Xp')') - P.R_t1).^2)));

%% Función: retropropagación con regla delta generalizada (por patrón, con momento)
function [hc,hv,bW1,bW2,kbest] = entrenarMLP(Xc,dc,Xv,dv,H,eta,alfa,nEp,semilla)
r  = RandStream('twister','Seed',semilla);
m  = size(Xc,2); n = size(Xc,1);
W1 = -0.5 + rand(r,H,m+1);  W2 = -0.5 + rand(r,1,H+1);   % pesos [sesgo, entradas]
dW1 = zeros(size(W1)); dW2 = zeros(size(W2));
sig = @(z) 1./(1+exp(-z));
mse = @(W1,W2,X,d) mean((d - sig([ones(size(X,1),1) sig([ones(size(X,1),1) X]*W1')]*W2')).^2);
hc = zeros(nEp+1,1); hv = hc;
hc(1) = mse(W1,W2,Xc,dc); hv(1) = mse(W1,W2,Xv,dv);
bW1 = W1; bW2 = W2; kbest = 0;
for ep = 1:nEp
    for p = randperm(r,n)
        x  = [1; Xc(p,:)'];                 % entrada con sesgo
        h  = sig(W1*x);                     % capa oculta (H x 1)
        hb = [1; h];
        y  = sig(W2*hb);                    % salida
        do = (dc(p) - y)*y*(1 - y);         % δ de salida
        dh = h.*(1 - h).*(W2(2:end)'*do);   % δ de la capa oculta (retropropagado)
        dW2 = eta*do*hb' + alfa*dW2;   W2 = W2 + dW2;
        dW1 = eta*dh*x'  + alfa*dW1;   W1 = W1 + dW1;
    end
    hc(ep+1) = mse(W1,W2,Xc,dc); hv(ep+1) = mse(W1,W2,Xv,dv);
    if hv(ep+1) < hv(kbest+1), kbest = ep; bW1 = W1; bW2 = W2; end
end
end
