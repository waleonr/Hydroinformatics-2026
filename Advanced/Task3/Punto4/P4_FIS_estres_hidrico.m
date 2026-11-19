%% PUNTO 4 — FIS para un problema técnico: riesgo de estrés hídrico del maíz en Tibaitatá
% Sitio: C.I. Tibaitatá (AGROSAVIA), Mosquera, Cundinamarca. Cultivo: maíz, etapa media.
%
% Organizado en secciones (%%) para abrirlo como Live Script
% (clic derecho > "Open as Live Script" y guardar como .mlx).
% Requisitos: MATLAB R2019b o posterior + Fuzzy Logic Toolbox.
% ESTADO: código propuesto. Las cifras del informe provienen de una réplica en
% Python; deben confirmarse al ejecutar este script en MATLAB.

clear; clc; close all; rng(2026,'twister');

%% 1. Definición del sistema
% Salida: riesgo de estrés hídrico al cierre del día t+1, R = 1 - Ks(Dr_{t+1}).
% Ks es el coeficiente de estrés hídrico de FAO-56 (Allen et al., 1998):
%   Ks = 1                               si Dr <= RAW = p*TAW
%   Ks = (TAW - Dr) / ((1 - p) * TAW)    si Dr >  RAW
% Entradas:
%   fD  = Dr_t / TAW : agotamiento relativo de la zona radicular al cierre del día t
%                      (equivale a la humedad del suelo: fD = (thFC - th)/(thFC - thWP))
%   ET0 del día t+1  : demanda evaporativa (pronóstico)
%   P   del día t+1  : lluvia (pronóstico)
% Supuesto de "pronóstico perfecto": ET0 y P de t+1 se tratan como conocidos.

% Parámetros del suelo y del cultivo (SUPUESTOS declarados; ajustar con datos de Tibaitatá)
par.thFC = 0.38;  par.thWP = 0.22;  par.Zr = 1.0;     % franco arcilloso (FAO-56, tabla 19); Zr en m
par.TAW  = 1000*(par.thFC - par.thWP)*par.Zr;           % 160 mm
par.Kc   = 1.20;  par.ptab = 0.55;                      % maíz, etapa media (FAO-56, tablas 12 y 22)

%% 2. FIS principal: Sugeno de orden cero, AND = producto, promedio ponderado
% Se eligió frente a Mamdani porque:
%  (a) con particiones de Ruspini, AND = producto y una tabla de reglas monótona,
%      la salida es una interpolación multilineal de la tabla: monótona por construcción;
%  (b) puede valer exactamente 0 (Mamdani con centroide nunca baja de ~0,09);
%  (c) es la misma familia que ANFIS (punto 7).
fis = sugfis('Name','RiesgoEstresMaiz','AndMethod','prod','DefuzzificationMethod','wtaver');

fis = addInput(fis,[0 1],'Name','fD');
fis = addMF(fis,'fD','trapmf',[0 0 0.40 0.55],   'Name','Holgado');
fis = addMF(fis,'fD','trimf', [0.40 0.55 0.70],  'Name','Cercano');   % alrededor del umbral p
fis = addMF(fis,'fD','trimf', [0.55 0.70 0.85],  'Name','Deficit');
fis = addMF(fis,'fD','trapmf',[0.70 0.85 1 1],   'Name','Critico');

fis = addInput(fis,[1 6],'Name','ET0');
fis = addMF(fis,'ET0','trapmf',[1 1 2 3],    'Name','Baja');
fis = addMF(fis,'ET0','trimf', [2 3 4.5],    'Name','Media');
fis = addMF(fis,'ET0','trapmf',[3 4.5 6 6],  'Name','Alta');

fis = addInput(fis,[0 30],'Name','P');
fis = addMF(fis,'P','trapmf',[0 0 1 5],     'Name','Nula');
fis = addMF(fis,'P','trimf', [1 5 12],      'Name','Ligera');
fis = addMF(fis,'P','trapmf',[5 12 30 30],  'Name','Fuerte');

fis = addOutput(fis,[0 1],'Name','Riesgo');
fis = addMF(fis,'Riesgo','constant',0.00,'Name','Nulo');
fis = addMF(fis,'Riesgo','constant',0.25,'Name','Bajo');
fis = addMF(fis,'Riesgo','constant',0.45,'Name','Moderado');
fis = addMF(fis,'Riesgo','constant',0.65,'Name','Alto');
fis = addMF(fis,'Riesgo','constant',0.85,'Name','Severo');

%% 3. Base de reglas (36 = 4 x 3 x 3)
% T{fD}(P, ET0): 1 = Nulo ... 5 = Severo. Columnas: ET0 Baja, Media, Alta.
T = cell(1,4);
T{1} = [1 1 1; 1 1 1; 1 1 1];   % Holgado: el riesgo es nulo en cualquier caso
T{2} = [1 1 2; 1 1 1; 1 1 1];   % Cercano: solo hay riesgo con demanda alta y sin lluvia
T{3} = [1 2 3; 1 2 3; 1 1 2];   % Déficit: la demanda mueve el umbral p
T{4} = [4 5 5; 4 4 4; 2 3 4];   % Crítico: solo la lluvia fuerte reduce el riesgo
R = zeros(36,6); k = 0;
for f = 1:4
  for e = 1:3
    for p = 1:3
      k = k + 1;
      R(k,:) = [f e p T{f}(p,e) 1 1];
    end
  end
end
fis = addRule(fis,R);

%% 4. Variante Mamdani con las mismas reglas (para comparar topologías)
fisM = mamfis('Name','RiesgoEstresMaiz_Mamdani','AndMethod','min', ...
              'ImplicationMethod','min','AggregationMethod','max','DefuzzificationMethod','centroid');
for i = 1:3
  fisM = addInput(fisM, fis.Inputs(i).Range, 'Name', fis.Inputs(i).Name);
  for j = 1:numel(fis.Inputs(i).MembershipFunctions)
    mf = fis.Inputs(i).MembershipFunctions(j);
    fisM = addMF(fisM, fis.Inputs(i).Name, mf.Type, mf.Parameters, 'Name', mf.Name);
  end
end
fisM = addOutput(fisM,[0 1],'Name','Riesgo');
fisM = addMF(fisM,'Riesgo','trapmf',[0 0 0.10 0.25],   'Name','Nulo');
fisM = addMF(fisM,'Riesgo','trimf', [0.10 0.25 0.45],  'Name','Bajo');
fisM = addMF(fisM,'Riesgo','trimf', [0.25 0.45 0.65],  'Name','Moderado');
fisM = addMF(fisM,'Riesgo','trimf', [0.45 0.65 0.85],  'Name','Alto');
fisM = addMF(fisM,'Riesgo','trapmf',[0.65 0.85 1 1],   'Name','Severo');
fisM = addRule(fisM,R);

%% 5. Funciones de pertenencia y superficies de respuesta
figure('Name','Funciones de pertenencia');
subplot(1,3,1); plotmf(fis,'input',1); title('f_D = D_r/TAW (día t)');
subplot(1,3,2); plotmf(fis,'input',2); title('ET_0 día t+1 (mm/d)');
subplot(1,3,3); plotmf(fis,'input',3); title('Lluvia día t+1 (mm)');

figure('Name','Superficies');
subplot(2,2,1); gensurf(fis, gensurfOptions('InputIndex',[1 2],'ReferenceInputs',[NaN NaN 0]));
title('Sugeno: f_D x ET_0, sin lluvia');
subplot(2,2,2); gensurf(fis, gensurfOptions('InputIndex',[1 3],'ReferenceInputs',[NaN 3 NaN]));
title('Sugeno: f_D x lluvia, ET_0 = 3');
subplot(2,2,3); gensurf(fisM, gensurfOptions('InputIndex',[1 2],'ReferenceInputs',[NaN NaN 0]));
title('Mamdani: f_D x ET_0, sin lluvia');
subplot(2,2,4); gensurf(fisM, gensurfOptions('InputIndex',[1 3],'ReferenceInputs',[NaN 3 NaN]));
title('Mamdani: f_D x lluvia, ET_0 = 3');

%% 6. Coherencia física: comparación con el balance FAO-56 en 5.000 combinaciones
n = 5000;
X = [rand(n,1), 1 + 5*rand(n,1), 30*rand(n,1)];
yRef = riesgoFAO56(X(:,1), X(:,2), X(:,3), par);
yS = evalfis(fis, X);  yM = evalfis(fisM, X);
met = @(y) [sqrt(mean((y-yRef).^2)), mean(abs(y-yRef)), mean(y-yRef), ...
            1 - sum((y-yRef).^2)/sum((yRef-mean(yRef)).^2)];
disp(array2table([met(yS); met(yM)], 'VariableNames',{'RMSE','MAE','Sesgo','R2'}, ...
     'RowNames',{'Sugeno0','Mamdani'}))
% Réplica Python: Sugeno RMSE 0,077, sesgo +0,008, R2 0,896;
%                 Mamdani RMSE 0,122, sesgo +0,084, R2 0,738

figure('Name','FIS vs FAO-56');
subplot(1,2,1); scatter(yRef,yS,6,'filled','MarkerFaceAlpha',0.3); refline(1,0);
xlabel('Riesgo FAO-56'); ylabel('Riesgo FIS'); title('Sugeno 0'); axis([0 1 0 1]);
subplot(1,2,2); scatter(yRef,yM,6,'filled','MarkerFaceAlpha',0.3); refline(1,0);
xlabel('Riesgo FAO-56'); title('Mamdani'); axis([0 1 0 1]);

% Monotonía en una malla 41 x 21 x 21
[A,B,C] = ndgrid(linspace(0,1,41), linspace(1,6,21), linspace(0,30,21));
Z = reshape(evalfis(fis,[A(:) B(:) C(:)]), size(A));
fprintf('Mayor descenso al subir fD: %.4f | al subir ET0: %.4f | mayor aumento al subir P: %.4f\n', ...
    max(-diff(Z,1,1),[],'all'), max(-diff(Z,1,2),[],'all'), max(diff(Z,1,3),[],'all'));
% Réplica Python: 0 | 0 | 0 (Sugeno producto). Mamdani: 0,021 | 0,037 | 0,053

%% 7. Escenarios de mayor y menor riesgo
esc = [0.10 5.5 0; 0.55 1.5 0; 0.55 5.5 0; 0.70 3 0; 0.70 3 20; 0.90 5.5 0; 0.90 1.5 25; 1.00 3 0];
disp(table(esc(:,1),esc(:,2),esc(:,3), riesgoFAO56(esc(:,1),esc(:,2),esc(:,3),par), ...
     evalfis(fis,esc), evalfis(fisM,esc), ...
     'VariableNames',{'fD','ET0','P','FAO56','Sugeno','Mamdani'}))

%% Funciones locales
function r = riesgoFAO56(fD, et0, P, par)
% Riesgo de referencia = 1 - Ks al cierre del día t+1 (balance FAO-56, sin escorrentía)
etc = par.Kc .* et0;
p   = min(max(par.ptab + 0.04*(5 - etc), 0.1), 0.8);   % ajuste de p por demanda (FAO-56)
dr  = fD .* par.TAW;
ks0 = ksFAO(dr, p, par.TAW);                            % Ks del día t+1 (usa Dr_t)
dr1 = min(max(dr - P + ks0.*etc, 0), par.TAW);           % Dr al cierre del día t+1
r   = 1 - ksFAO(dr1, p, par.TAW);
end

function ks = ksFAO(dr, p, TAW)
ks = (TAW - dr) ./ ((1 - p) .* TAW);
ks(dr <= p.*TAW) = 1;
ks = min(max(ks,0),1);
end
