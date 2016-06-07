addpath(genpath('C:\Users\aj526\Downloads\mldscomp2\HMM\HMMall'))
data = csvread('fillindata.csv',0,0,[0 0 99 99]);
pred = csvread('hmm.csv',100,0,[100 0 999 99]);

% 
for r=1:length(pred)
    idx = find(pred(r,:) == 0);
    pred(r,idx) = out(r);
end
% %
S = 7;
K = 5;

[estr, esem] = hmmtrain(pred,estrKM,esemKM);
% 
% rng(46);
% trguess = rand(S);
% trguess = trguess./repmat(sum(trguess,2),1,S);
% 
% %emguess = [0.4,0.1,0.1,0.1,0.3;0.4,0.1,0.1,0.1,0.3;];
% prior = normalize(rand(S,1));
% emguess = [0.4,0.1,0.1,0.1,0.3];
% emguess = repmat(emguess,S,1);
% 
% % Train
% %[LL, priorl, estrKM,esemKM] = dhmm_em(data(1:99,:), prior, trguess, emguess, 'thresh', 1e-6, 'verbose',1, 'max_iter',500)
% [LL, priorl, estrKM,esemKM] = dhmm_em(data, prior, trguess, emguess, 'thresh', 1e-6, 'verbose',0, 'max_iter',500);
% 
% % Infer
% % tmppred = data(100,:);
% % correct = tmppred(1,65);
% % for i=1:5
% %     tmppred(1,65) = i;
% %     [PSTATES,logpseq] = hmmdecode(tmppred,estrKM,esemKM)
% %     marr(i) = max(PSTATES(:,65));
% % end

% Infer
out = zeros(900,1);
for r=1:length(pred)
    maxarr = zeros(1,K);
    idx = find(pred(r,:) == 0);
    for i=1:K
        pred(r,idx) = i;
        [PSTATES,logpseq] = hmmdecode(pred(r,:),estrKM,esemKM);
        maxarr(i) = logpseq;
    end
    [~,out(r)] = max(maxarr);
end

for r=1:length(data)
    disp(r)
    [a,b] = hmmdecode(data(r,:),estrKM,esemKM);
    b
end

ll = zeros(900,1);
for r=1:length(pred)
    idx = find(pred(r,:) == 0);
    pred(r,idx) = out(r);
    [a,b] = hmmdecode(pred(r,:),estrKM,esemKM);
    ll(r,1) = b;
end
