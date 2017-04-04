
require 'torch'   -- torch
matio = require 'matio'


--model = torch.load('mul-class/log_exp_64_256_64_.5_.1_10class_r_sig_nll/'..'model.net')


direc = '/BS/deep_3d/work/deep_3d/fcn_3D/Data/'

 i, t, popen = 0, {}, io.popen
     pfile = popen('ls -a "'..direc..'"')
    for filename in pfile:lines() do
        i = i + 1
        t[i] = filename
	print(filename)
    end
    pfile:close()

--local p = io.popen('find "'..dir..'" -type f')  --Open directory look for files, save data in p. By giving '-type f' as parameter, it returns all files.     
   --for file in p:lines() do                         --Loop through all files
       --print(file)       
   --end
