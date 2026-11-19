%% PUNTO 5 — Red neuronal artificial SIN capa oculta (perceptrón simple, salida sigmoide)
% Hidroinformática Avanzada · Tarea 3 · Autor: Douglas A. Gómez-Latorre
% Problema: riesgo de estrés hídrico del maíz en Tibaitatá, R_{t+1} = 1 - Ks (FAO-56)
% Entradas: fD_t, ET0_{t+1}, P_{t+1}.  Archivos: tabla_patrones_Zr1.0.csv,
% serie_diaria_procesada_Zr1.0.csv (solo para el experimento de sobreentrenamiento).
%
% Regla de clase: error cuadrático medio, activación sigmoide,
%   Δw = η·(d − y)·f'(net)·x, con actualización después de CADA patrón.
% No usa toolboxes: el entrenamiento se programa explícitamente.
% ESTADO: código propuesto. Las cifras del informe provienen de una réplica en Python
% con el mismo algoritmo; los números aleatorios de MATLAB darán valores ligeramente distintos.

clear; clc; close all; rng(2026,'twister');

%% 1. Tabla de patrones y partición cronológica
T  = readtable('tabla_patrones_Zr1.0.csv');
IN = {'fD_t','ET0_t1_mm','P_t1_mm'};
C = T(strcmp(T.conjunto,'calibracion'),:);   % 2007-2015
V = T(strcmp(T.conjunto,'validacion'),:);    % 2016-2017
P = T(strcmp(T.conjunto,'prueba'),:);        % 2018-2020

%% 2. Normalización (ajustada SOLO con calibración)
% Entradas a [0,1]; salida a [0.1, 0.9] para no trabajar en la zona saturada de la sigmoide.
lo = min(C{:,IN});  hi = max(C{:,IN});  dlo = min(C.R_t1);  dhi = max(C.R_t1);
nx  = @(X) (X - lo)./(hi - lo);
nd  = @(d) 0.1 + 0.8*(d - dlo)/(dhi - dlo);
invd = @(y) dlo + (y - 0.1)/0.8*(dhi - dlo);
Xc = nx(C{:,IN});  Xv = nx(V{:,IN});  Xp = nx(P{:,IN});
dc = nd(C.R_t1);   dv = nd(V.R_t1);   dp = nd(P.R_t1);

%% 3. Formulación matemática (ver informe)
% net = w0 + Σ w_i x_i ;  y = σ(net) = 1/(1+e^{-net}) ;  σ'(net) = y(1−y)
% Error del patrón p: E_p = ½ (d_p − y_p)^2 ; error de la época: MSE = (1/N) Σ (d_p − y_p)^2
% ∂E_p/∂w_i = −(d_p − y_p)·y_p(1−y_p)·x_i   =>   Δw_i = η (d_p − y_p) y_p (1−y_p) x_i

%% 4. Selección de la tasa de aprendizaje con el conjunto de VALIDACIÓN
etas = [0.01 0.05 0.1 0.5 1.0];  nEp = 150;  semillas = 1:5;
best = zeros(numel(etas),1);
for a = 1:numel(etas)
    v = zeros(numel(semillas),1);
    for s = semillas
        [~,hv] = entrenarSLP(Xc,dc,Xv,dv,etas(a),nEp,s,[]);
        v(s) = min(hv);
    end
    best(a) = mean(v);
    fprintf('eta = %.2f -> MSE de validación mínimo medio = %.5f\n', etas(a), best(a));
end
% Réplica Python: 0,01 -> 0,00351; 0,05 -> 0,00100; 0,1 / 0,5 / 1,0 -> 0,00096.
% Se elige η = 0,1: el menor error de validación con convergencia estable.

%% 5. Entrenamiento final con parada temprana (época de menor error de validación)
eta = 0.1;  nEp = 300;
[hc,hv,W] = entrenarSLP(Xc,dc,Xv,dv,eta,nEp,1,[]);
[~,k] = min(hv);  w = W(k,:)';
fprintf('Época elegida: %d | pesos [w0 fD ET0 P] = %s\n', k-1, mat2str(w',3));
% Réplica Python: época 271; pesos ≈ [-7.46 8.83 0.87 -3.83]

figure('Name','Curvas de aprendizaje');
semilogy(0:nEp, hc,'LineWidth',1.5); hold on; semilogy(0:nEp, hv,'LineWidth',1.5);
xline(k-1,'--',sprintf('época %d',k-1)); xlabel('Época'); ylabel('MSE (normalizado)');
legend('Calibración','Validación'); title('Curvas de aprendizaje, η = 0,1');

%% 6. Desempeño en escala original y gráficas de dispersión
pred = @(X) invd(1./(1+exp(-[ones(size(X,1),1) X]*w)));   % w ya es el vector final
conj = {'Calibración',C,Xc; 'Validación',V,Xv; 'Prueba',P,Xp};
figure('Name','Observado vs predicho');
for j = 1:3
    o = conj{j,2}.R_t1;  yhat = pred(conj{j,3});  e = yhat - o;
    m = [sqrt(mean(e.^2)), mean(abs(e)), mean(e), 1 - sum(e.^2)/sum((o-mean(o)).^2)];
    fprintf('%-12s RMSE %.3f | MAE %.3f | sesgo %+.3f | R2 %.3f\n', conj{j,1}, m);
    subplot(1,3,j); scatter(o,yhat,6,'filled','MarkerFaceAlpha',0.3); hold on;
    plot([0 1],[0 1],'k--'); axis([0 1 0 1]); axis square;
    xlabel('R observado (FAO-56)'); ylabel('R predicho');
    title(sprintf('%s: RMSE = %.3f, R^2 = %.3f', conj{j,1}, m(1), m(4)));
end
% Réplica Python: prueba RMSE 0,037 | MAE 0,024 | sesgo -0,009 | R2 0,984
% (FIS Sugeno del punto 4 en la misma prueba: RMSE 0,084, R2 0,919)

%% 7. Experimento de sobreentrenamiento
% Condiciones para que aparezca: pocos patrones de calibración (15) frente a muchos
% pesos (13: 12 entradas con retardos redundantes + sesgo). Control: las 3 entradas originales.
Q  = readtable('serie_diaria_procesada_Zr1.0.csv');  Q.Properties.VariableNames{1} = 'Fecha';
[~,ia] = ismember(T.fecha_t, Q.Fecha);
lagv = @(col,k) Q.(col)(max(ia-k,1));
Text = T;
Text.fD_1 = lagv('fD',1); Text.fD_2 = lagv('fD',2); Text.fD_3 = lagv('fD',3);
Text.P_0  = lagv('P',0);  Text.P_1  = lagv('P',1);  Text.P_2  = lagv('P',2);
Text.E_0  = lagv('et0',0); Text.E_1 = lagv('et0',1); Text.E_2 = lagv('et0',2);
IN12 = [IN, {'fD_1','fD_2','fD_3','P_0','P_1','P_2','E_0','E_1','E_2'}];
Text = Text(all(~isnan(Text{:,IN12}),2),:);
Ce = Text(strcmp(Text.conjunto,'calibracion'),:);  Ve = Text(strcmp(Text.conjunto,'validacion'),:);

figure('Name','Sobreentrenamiento');
for caso = 1:2
    if caso == 1, cols = IN12; tit = '12 entradas, 15 patrones'; else, cols = IN; tit = '3 entradas, 15 patrones (control)'; end
    subplot(1,2,caso); hold on;
    for s = 1:4
        r = RandStream('twister','Seed',s);  sub = Ce(randperm(r,height(Ce),15),:);
        lo2 = min(sub{:,cols}); hi2 = max(sub{:,cols}); a2 = min(sub.R_t1); b2 = max(sub.R_t1);
        Xs = (sub{:,cols}-lo2)./(hi2-lo2);  Xvs = (Ve{:,cols}-lo2)./(hi2-lo2);
        ds = 0.1+0.8*(sub.R_t1-a2)/(b2-a2); dvs = 0.1+0.8*(Ve.R_t1-a2)/(b2-a2);
        [hcs,hvs] = entrenarSLP(Xs,ds,Xvs,dvs,2.0,20000,0,zeros(numel(cols)+1,1));
        loglog(1:numel(hvs),hvs,'-'); loglog(1:numel(hcs),hcs,':');
        [mn,kk] = min(hvs);
        fprintf('%s | muestra %d: mínimo de validación en época %d; validación final +%.0f %%\n', tit, s, kk-1, 100*(hvs(end)/mn-1));
    end
    set(gca,'XScale','log','YScale','log'); xlabel('Época'); ylabel('MSE'); title(tit);
end
% Réplica Python (12 entradas): validación final +69 %, +69 %, +1.099 %, +86 % sobre su mínimo.
% Control (3 entradas): +4 %, +1.772 %, +23 %, +1 %.

%% Función de entrenamiento: regla delta, actualización por patrón
function [hc,hv,W] = entrenarSLP(Xc,dc,Xv,dv,eta,nEp,semilla,w0)
r  = RandStream('twister','Seed',semilla);
n  = size(Xc,1);  Xc1 = [ones(n,1) Xc];  Xv1 = [ones(size(Xv,1),1) Xv];
if isempty(w0), w = -0.5 + rand(r,size(Xc1,2),1); else, w = w0; end
sig = @(z) 1./(1+exp(-z));
% Ojo: una función anónima congela el valor de w al crearse; por eso se pasa w como argumento
mse = @(X1,d,w) mean((d - sig(X1*w)).^2);
hc = zeros(nEp+1,1); hv = hc; W = zeros(nEp+1,numel(w));
hc(1) = mse(Xc1,dc,w); hv(1) = mse(Xv1,dv,w); W(1,:) = w';
for ep = 1:nEp
    for i = randperm(r,n)                       % orden aleatorio de patrones en cada época
        x = Xc1(i,:)';  y = sig(w'*x);
        w = w + eta*(dc(i) - y)*y*(1 - y)*x;     % Δw = η (d − y) f'(net) x
    end
    hc(ep+1) = mse(Xc1,dc,w); hv(ep+1) = mse(Xv1,dv,w); W(ep+1,:) = w';
end
end
