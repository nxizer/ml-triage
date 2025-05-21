# ml-triage

## Clone project repository and set environment variables
```
cd <workdir>
git clone https://github.com/nxizer/ml-triage.git
cd ml-triage
export $ML_TRIAGE_HOME=<workdir>/ml-triage
```

## Requirements

## Obtain SARD data
To run the model and other scripts you will need the NIST SARD testcases data,  
which can be obtained three ways:
1.  Run `./sard_testcases.sh` to automatically download and extract archive to the right place
    ```
    sh sard_testcases.sh
    ```
2.  Manually download tar.gz archive from this [link](https://disk.yandex.ru/d/JBuGu9n-DZTCFA)
    and extract it to ./dataset/
3.  Run `./utils/sard_api.py` python scrypt to obtain all the data directly from NIST SARD API