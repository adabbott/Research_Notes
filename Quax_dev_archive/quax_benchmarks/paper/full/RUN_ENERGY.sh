# First time this is run, all integrals will be written. Second time integrals will not be written
cd n2/
cd dz/
cd energy/
echo "N2 cc-pVDZ energy"
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore ccsdt_1.py
cd ..
cd ..
cd tz/
cd energy/
echo "N2 cc-pVTZ energy"
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore ccsdt_1.py
cd ..
cd ..
cd ..

cd h2o/
cd dz/
cd energy/
echo "H2O cc-pVDZ energy"
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore ccsdt_1.py
cd ..
cd ..
cd tz/
cd energy/
echo "H2O cc-pVTZ energy"
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore ccsdt_1.py
cd ..
cd ..
cd ..

cd h2co/
cd dz/
cd energy/
echo "H2CO cc-pVDZ energy"
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore ccsdt_1.py
cd ..
cd ..
cd tz/
cd energy/
echo "H2CO cc-pVTZ energy"
/usr/bin/time python -W ignore hf_1.py
/usr/bin/time python -W ignore mp2_1.py
/usr/bin/time python -W ignore ccsdt_1.py
cd ..
cd ..
cd ..

