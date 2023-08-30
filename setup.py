from setuptools import setup, find_packages



setup(
        name='kernelnb',
        version ='0.0.1',
        author='Pawel Trajdos',
        author_email='pawel.trajdos@pwr.edu.pl',
        url = 'https://github.com/ptrajdos/KernelNB',
        description="Naive bayes with kernel estimators",
        packages=find_packages(include=[
                'kernelnb',
                'kernelnb.*',
                ]),
        install_requires=[ 
                'numpy==1.23.4',
                'joblib',
                'scikit-learn>=1.2.2',
        ],
        test_suite='test'
        )
