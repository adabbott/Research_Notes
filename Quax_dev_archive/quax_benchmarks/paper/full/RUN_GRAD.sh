# First time this is run, all integrals will be written. Second time integrals will not be written
cd n2/
cd dz/
cd grad/
echo "N2 cc-pVDZ gradients"
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore ccsdt_1.py
cd ..
cd ..
cd tz/
cd grad/
echo "N2 cc-pVTZ gradients"
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore ccsdt_1.py
cd ..
cd ..
cd ..

cd h2o/
cd dz/
cd grad/
echo "H2O cc-pVDZ gradients"
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore ccsdt_1.py
cd ..
cd ..
cd tz/
cd grad/
echo "H2O cc-pVTZ gradients"
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore ccsdt_1.py
cd ..
cd ..
cd ..

cd h2co/
cd dz/
cd grad/
echo "H2CO cc-pVDZ gradients"
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore ccsdt_1.py
cd ..
cd ..
cd tz/
cd grad/
echo "H2CO cc-pVTZ gradients"
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore ccsdt_1.py
cd ..
cd ..
cd ..

