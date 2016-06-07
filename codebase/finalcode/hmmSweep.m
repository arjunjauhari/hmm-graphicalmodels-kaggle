clear;
data = csvread('fillindata.csv',0,0,[0 0 99 99]);
%
for iter=1:11
    S = iter;
    K = 5;

    rng(46);
    trguess = rand(S);
    trguess = trguess./repmat(sum(trguess,2),1,S);

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
        % Validation
        for s=st:en
            display(s)
            [~, valLL(s,1)] = hmmdecode(data(s,:),estr,esem);
        end
        mean(valLL(st:en))
    end
    disp(['Final: ' num2str(S) ' : ' num2str(mean(valLL))])
end

