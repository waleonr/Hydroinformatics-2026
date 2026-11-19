%% PUNTO 3 — FIS para una situación personal: ¿con cuánta antelación salgo hacia la oficina?
% Hidroinformática Avanzada ·
% Trayecto: Chía (Cundinamarca) -> Mosquera (Cundinamarca), salida por la mañana.
%
% Organizado en secciones (%%) para abrirlo como Live Script:
% clic derecho > "Open as Live Script" y guardar como .mlx.
% Requisitos: MATLAB R2019b o posterior + Fuzzy Logic Toolbox.
% ESTADO: código propuesto. Las cifras del informe provienen de una réplica en
% Python (Mamdani min-max, centroide con 101 puntos); deben confirmarse en MATLAB.

clear; clc; close all;

%% 1. Enfoque y topología
% Enfoque DESCRIPTIVO: el sistema imita mi comportamiento real, no la decisión prudente.
% Topología: Mamdani, 3 entradas, 1 salida, 27 reglas (malla completa 3x3x3).
% Se elige Mamdani y no Sugeno porque los consecuentes son lingüísticos
% ("salgo con mucha antelación"), lo que facilita escribir y auditar las reglas.

fis = mamfis('Name','AntelacionSalida', ...
    'AndMethod','min','OrMethod','max', ...
    'ImplicationMethod','min','AggregationMethod','max', ...
    'DefuzzificationMethod','centroid');

%% 2. Variables lingüísticas y funciones de pertenencia
% Entrada 1: probabilidad de lluvia del pronóstico (app del tiempo), 0-100 %
fis = addInput(fis,[0 100],'Name','Lluvia');
fis = addMF(fis,'Lluvia','trapmf',[0 0 20 50],   'Name','SinLluvia');
fis = addMF(fis,'Lluvia','trimf', [20 50 80],    'Name','AlgunaLluvia');
fis = addMF(fis,'Lluvia','trapmf',[50 80 100 100],'Name','Lluvia');

% Entrada 2: congestión según el día. 1 = domingo, 2 = sábado (supuesto),
% 3 = martes, miércoles (supuesto) y jueves, 5 = lunes o viernes.
fis = addInput(fis,[1 5],'Name','Dia');
fis = addMF(fis,'Dia','trapmf',[1 1 2 3],'Name','Tranquilo');
fis = addMF(fis,'Dia','trimf', [2 3 5],  'Name','Normal');
fis = addMF(fis,'Dia','trapmf',[3 5 5 5],'Name','Pesado');

% Entrada 3: horas de sueño la noche anterior, 4-9 h
fis = addInput(fis,[4 9],'Name','Sueno');
fis = addMF(fis,'Sueno','trapmf',[4 4 5 6.5],'Name','DormiMal');
fis = addMF(fis,'Sueno','trimf', [5 6.5 8],  'Name','Normal');
fis = addMF(fis,'Sueno','trapmf',[6.5 8 9 9],'Name','Bien');

% Salida: minutos de antelación respecto a la hora de entrada, 30-130 min
fis = addOutput(fis,[30 130],'Name','Antelacion');
fis = addMF(fis,'Antelacion','trapmf',[30 30 40 55],    'Name','MuyPoca');
fis = addMF(fis,'Antelacion','trimf', [40 55 70],       'Name','Poca');
fis = addMF(fis,'Antelacion','trimf', [55 70 85],       'Name','Moderada');
fis = addMF(fis,'Antelacion','trimf', [70 85 100],      'Name','Amplia');
fis = addMF(fis,'Antelacion','trimf', [85 100 115],     'Name','MuyAmplia');
fis = addMF(fis,'Antelacion','trapmf',[100 115 130 130],'Name','Maxima');

%% 3. Base de reglas (27)
% Nivel base con sueño normal o bueno (1 = MuyPoca ... 6 = Maxima):
%                Tranquilo  Normal  Pesado
% SinLluvia          2        2       3
% AlgunaLluvia       3        4       5
% Lluvia             5        5       6
% La lluvia sube hasta 3 niveles; el día, 1 o 2: la lluvia pesa más (expectativa 4).
% Dormir mal resta un nivel (expectativa 3); dormir bien no cambia nada (supuesto).
base = [2 2 3; 3 4 5; 5 5 6];
R = zeros(27,6); k = 0;
for r = 1:3
  for d = 1:3
    for s = 1:3
      k = k + 1;
      out = base(r,d) - (s==1);
      R(k,:) = [r d s out 1 1];   % [lluvia dia sueno salida peso AND]
    end
  end
end
fis = addRule(fis,R);
showrule(fis)

%% 4. Funciones de pertenencia
figure('Name','Funciones de pertenencia');
subplot(2,2,1); plotmf(fis,'input',1);  title('Lluvia (%)');
subplot(2,2,2); plotmf(fis,'input',2);  title('Día (congestión 1-5)');
subplot(2,2,3); plotmf(fis,'input',3);  title('Sueño (h)');
subplot(2,2,4); plotmf(fis,'output',1); title('Antelación (min)');

%% 5. Superficies de respuesta
% gensurf fija la entrada que no se grafica en el punto medio de su rango;
% por eso se usa gensurfOptions para fijarla en valores con sentido.
figure('Name','Superficies');
subplot(1,3,1);
gensurf(fis, gensurfOptions('InputIndex',[1 2],'ReferenceInputs',[NaN NaN 7]));
title('Lluvia x día, sueño = 7 h');
subplot(1,3,2);
gensurf(fis, gensurfOptions('InputIndex',[1 2],'ReferenceInputs',[NaN NaN 4.5]));
title('Lluvia x día, sueño = 4,5 h');
subplot(1,3,3);
gensurf(fis, gensurfOptions('InputIndex',[1 3],'ReferenceInputs',[NaN 5 NaN]));
title('Lluvia x sueño, lunes/viernes');

%% 6. Contraste con las expectativas (escritas antes de ver la superficie)
casos = [100 5 7;    % E1: lluvia segura, día pesado, sueño normal  -> esperado ~120
           0 1 7;    % E2: sin lluvia, domingo, sueño normal        -> esperado ~50
          50 3 4.5;  % E3: dormí mal                                -> esperado: menos antelación
          50 3 7];   %     mismo caso, sueño normal
y = evalfis(fis,casos);
disp(table(casos(:,1),casos(:,2),casos(:,3),y, ...
     'VariableNames',{'Lluvia_pct','Dia','Sueno_h','Antelacion_min'}))
% Réplica Python: 117,6 | 55,0 | 70,0 | 85,0 min

% E4: ¿pesa más la lluvia que el día? Rango medio de la salida en cada dirección
[L,D] = meshgrid(0:10:100, 1:0.4:5);
Z = reshape(evalfis(fis,[L(:) D(:) 7*ones(numel(L),1)]), size(L));
fprintf('Rango medio al variar la lluvia: %.1f min\n', mean(range(Z,2)));
fprintf('Rango medio al variar el día:    %.1f min\n', mean(range(Z,1)));
% Réplica Python: 45,9 min (lluvia) frente a 20,2 min (día)
