beta = h5read('param.h5','/beta');
gamma = h5read('param.h5','/gamma');
epsilon = h5read('param.h5','/epsilon');
bsh = h5read('param.h5','/bsh');
gsh = h5read('param.h5','/gsh');


loc = '/home/laurent/Dropbox/Research/PIML Julia/data/dataset_118_case_3.h5';

g_ref = h5read(loc,'/g');
b_ref = h5read(loc,'/b');
epsilon_ref = h5read(loc,'/epsilon');
bsh_ref = h5read(loc,'/bsh');
gsh_ref = h5read(loc,'/gsh');


id_ref = 1E4*epsilon_ref(:,1)+epsilon_ref(:,2);
id = 1E4*epsilon(:,1)+epsilon(:,2);

bs = [];
gs = [];

for i=1:length(id_ref)
    j = find(id == id_ref(i));
    if(~isempty(j))
        bs = [bs; b_ref(i) -exp(beta(j))]
        gs = [gs; g_ref(i) exp(gamma(j))]
    end
end