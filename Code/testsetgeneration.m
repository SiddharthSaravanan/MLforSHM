%TESTING DATASET GENERATION
x=0:0.01:7;

ph = 2*pi*rand;
y = sin(5*x + ph);
outpure = y;
outnoisy = awgn(y,13);

%frequency=5, low noise
for i=1:99
    ph = 2*pi*rand;
    y = sin(5*x + ph);
    outpure = [outpure;y];
    outnoisy = [outnoisy;awgn(y,1)];
end

%frequency=5, medium noise
for i=1:100
    ph = 2*pi*rand;
    y = sin(5*x + ph);
    outpure = [outpure;y];
    outnoisy = [outnoisy;awgn(y,5)];
end

%frequency=5, harsh noise
for i=1:100
    ph = 2*pi*rand;
    y = sin(5*x + ph);
    outpure = [outpure;y];
    outnoisy = [outnoisy;awgn(y,0.5)];
end

%frequency=8, low noise
for i=1:100
    ph = 2*pi*rand;
    y = sin(8*x + ph);
    outpure = [outpure;y];
    outnoisy = [outnoisy;awgn(y,1)];
end

%frequency=8, medium noise
for i=1:100
    ph = 2*pi*rand;
    y = sin(8*x + ph);
    outpure = [outpure;y];
    outnoisy = [outnoisy;awgn(y,5)];
end

%frequency=8, harsh noise
for i=1:100
    ph = 2*pi*rand;
    y = sin(8*x + ph);
    outpure = [outpure;y];
    outnoisy = [outnoisy;awgn(y,0.5)];
end

%frequency=12, low noise
for i=1:100
    ph = 2*pi*rand;
    y = sin(12*x + ph);
    outpure = [outpure;y];
    outnoisy = [outnoisy;awgn(y,1)];
end

%frequency=12, medium noise
for i=1:100
    ph = 2*pi*rand;
    y = sin(12*x + ph);
    outpure = [outpure;y];
    outnoisy = [outnoisy;awgn(y,5)];
end

%frequency=12, harsh noise
for i=1:100
    ph = 2*pi*rand;
    y = sin(12*x + ph);
    outpure = [outpure;y];
    outnoisy = [outnoisy;awgn(y,0.5)];
end


xlswrite('test_pure.xlsx',outpure);
xlswrite('test_noisy.xlsx',outnoisy);