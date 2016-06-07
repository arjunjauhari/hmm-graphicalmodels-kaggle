clear;
addpath(genpath('C:\Users\aj526\Downloads\mldscomp2\HMM\HMMall'))
data = csvread('fillindata.csv',0,0,[0 0 99 99]);

%
for iter=1:11
    S = iter;
    K = 5;

    rng(46);
    trguess = rand(S);
    trguess = trguess./repmat(sum(trguess,2),1,S);

    %emguess = [0.4,0.1,0.1,0.1,0.3;0.4,0.1,0.1,0.1,0.3;];
    prior = normalize(rand(S,1));
    emguess = [0.4,0.1,0.1,0.1,0.3];
    emguess = repmat(emguess,S,1);
    nfold = 10;
    valLL = zeros(100,1);
    for fold=1:nfold
        st = fold*nfold-(nfold-1);
        en = fold*nfold;
        valset = data(st:en,:);
        trset = data([1:st-1 en+1:100],:);
        % Training
        display(['=========== Running FOLD ',num2str(fold),' =============='])

        [estr, esem] = hmmtrain(trset,trguess,emguess,'Verbose',true,'Tolerance',1e-4);
        %[LL, priorl, estrKM,esemKM] = dhmm_em(trset, prior, trguess, emguess, 'thresh', 1e-6, 'verbose',0, 'max_iter',500);
        % Validation
        for s=st:en
            display(s)
            %hmmtrain(data(s,:),estr,esem,'Verbose',true,'Maxiterations',2)
            [~, valLL(s,1)] = hmmdecode(data(s,:),estr,esem);
            %valLL(s,1) = dhmm_logprob(data(s,:),priorl,estrKM,esemKM);
        end
        mean(valLL(st:en))
    end
    disp(['Final: ' num2str(S) ' : ' num2str(mean(valLL))])
end

