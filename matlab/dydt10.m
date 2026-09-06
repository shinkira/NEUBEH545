function f = dydt(t,y, commandTime, F0, Fa, option)
% y = [ \theta, \theta', \theta'',  Fm, Fm', Fp, Fp'];
% output f = dy/dt
% t in unit of s

mi = [ 0.677e-4;   %in g * sec^2 / deg
	2.16e-4;    %in g * sec^2 /deg, isometric saccade
	28.9e-4];   %with sled load
Rm = 0.072;     %in g * sec / deg, isometric saccade
Ke = 3.6;       %in g /deg
K1 = 2.06;      %in g /deg
R1 = 0.025;     %in g * sec / deg
K2 = 6.36;      %in g /deg
R2 = 1.81;      %in g * sec / deg
Ki = 15;         %in g /dge, isometric saccades

F0 = interp1(commandTime,F0,t);    %in g, active state tension
Fa = interp1(commandTime,Fa,t);

m = mi(1);
if(strcmp(option, 'isometric'))
	m = mi(2);
	Fa = -Ki * y(1);
	Fa1 = -Ki * y(2);
end;

if(strcmp(option, 'normal'))
	m = mi(1);
end;

if(strcmp(option, 'high inertial'))
	m = mi(3);
end;

if(strcmp(option, 'isotonic'))
	m = mi(1);
end;

f = [y(2) ;
	(y(3) + Fa - y(4))/m;
	(F0 - Rm * y(2) -y(3))*Ke/Rm;
	(R1*R2*(y(3) + Fa -y(4))/m + (R1*K2+R2*K1)*y(2) + K1*K2*y(1) - (K1+K2)*y(4))/(R1+R2);
	];
