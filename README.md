### Installing libsvm on Mac OSX 10.10 with Matlab 2015a:

1) Download the file xcode7_mexopts.zip, which you will find attached to this article. Extract the contents of xcode7_mexopts.zip to your Downloads directory. Extracting the ZIP file will create a directory in Downloads called xcode7_mexopts.

2) Open MATLAB R2015b. Navigate to MATLAB's preference directory by typing the following command in the command window:

```
>> cd( prefdir );
```

3) Using MATLAB's "Current Folder" browser, ensure that there are no XML files that begin with "mex_" (e.g. mex_C_maci64.xml). If such files exist, remove them from the preference directory.

4) Navigate to the MATLAB directory. In MATLAB, you can do so by entering the following command at the MATLAB command prompt:

```
>> cd( fullfile( matlabroot, 'bin', 'maci64', 'mexopts' ) );
```

5) Back up the original files in the mexopts directory. You can run the following MATLAB commands to back up the files:

```
>> mkdir mexoptsContentsOLD
>> movefile *.xml mexoptsContentsOLD/
```

Keep the backup separate from the downloaded files such that you can revert to the backup files if necessary.

6) Replace any files in the mexopts directory with the corresponding files in the Downloads folder, xcode7_mexopts. You can replace the files in MATLAB by entering the following command at the MATLAB command prompt:

```
>> movefile( '~/Downloads/xcode7_mexopts/*.xml', '.' );
```

7) Restart MATLAB R2015b. Execute "mex -setup" at the MATLAB command prompt as shown below and verify that MEX detects Xcode 7.

```
>> mex -setup
```

### Installing LibSVM 

1) Download from [LibSVM website (version 3.20 used for this project)](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

2) Install Xcode

3) Restart / Open Matlab

4) Navigate to `LIBSVM_FOLDER/libsvm-3.20/matlab/`

5) `make`

6) The `.mexmaci64` and `.mexw64` files for `libsvmread`, `libsvmwrite`, `svmtrain`, and `svmpredict` should be created.

7) Add libsvm-3.20/matlab/ to your path using the 'Set Path' button under the 'Home' tab on your Matlab GUI.
