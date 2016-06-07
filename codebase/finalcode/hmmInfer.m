data = csvread('fillindata.csv',0,0,[0 0 99 99]);
pred = csvread('hmm.csv',100,0,[100 0 999 99]);

% Finalized best hyper-parameters
S = 10;
K = 5;

rng(46);
trguess = rand(S);
trguess = trguess./repmat(sum(trguess,2),1,S);

emguess = [0.4,0.1,0.1,0.1,0.3];
emguess = repmat(emguess,S,1);

% Train
[estrKM, esemKM] = hmmtrain(data,trguess,emguess,'Verbose',true,'Tolerance',1e-4);

% Infer
out = zeros(900,1);
maxarr = zeros(900,K);
for r=1:length(pred)
    idx = find(pred(r,:) == 0);
    for i=1:K
        pred(r,idx) = i;
        [PSTATES,logpseq] = hmmdecode(pred(r,:),estrKM,esemKM);
        maxarr(r,i) = logpseq;
    end
end
[~,out] = max(maxarr,[],2);