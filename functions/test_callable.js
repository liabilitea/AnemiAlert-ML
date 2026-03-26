// test_callable.js
const { initializeApp } = require('firebase/app');
const { getFunctions, httpsCallable } = require('firebase/functions');

// Firebase config (Android app)
const firebaseConfig = {
  apiKey: "AIzaSyBnWZ7dGI-6hxhSeRMrtM6dLrsOgR7GomM",  
  authDomain: "anemialert-7b776.firebaseapp.com",
  databaseURL: "https://anemialert-7b776-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "anemialert-7b776",
  storageBucket: "anemialert-7b776.appspot.com",
  messagingSenderId: "543320714991",
  appId: "1:543320714991:android:8a44a8860d456543fa351a",
};

const app = initializeApp(firebaseConfig);
const functions = getFunctions(app, 'asia-southeast1');

async function testCallable() {
  try {
    console.log('Testing predict_hemoglobin_integrated...\n');
    
    const predict = httpsCallable(functions, 'predict_hemoglobin_integrated');
    
    console.log('Calling function with record: -OeL6Uudt-EREZD_KYkl');
    const startTime = Date.now();
    
    const result = await predict({ 
      record_id: '-OePZtNs1nh-SPFmXXLq'
    });
    
    const endTime = Date.now();
    const duration = ((endTime - startTime) / 1000).toFixed(2);
    
    console.log(`\nSUCCESS! (${duration} seconds)\n`);
    console.log('PREDICTION RESULTS:');
    console.log(`Hemoglobin Level: ${result.data.hemoglobin_prediction.toFixed(2)} g/dL`);
    console.log(`Anemia Status: ${result.data.anemia_status}`);
    console.log(`Eye-based Hb: ${result.data.eye_hemoglobin.toFixed(2)} g/dL`);
    console.log(`PPG-based Hb: ${result.data.ppg_hemoglobin.toFixed(2)} g/dL`);
    console.log(`Weights: ${(result.data.weights.eye * 100).toFixed(0)}% eye, ${(result.data.weights.ppg * 100).toFixed(0)}% PPG`);
    console.log(`Result saved to: /result/${result.data.result_id}`);
    
    if (result.data.segmentation_stats) {
      console.log(`Mask Coverage: ${result.data.segmentation_stats.mask_coverage_percent.toFixed(2)}%`);
    }
    
    console.log('\nFull Response:\n');
    console.log(JSON.stringify(result.data, null, 2));
    
  } catch (error) {
    console.log('\nERROR!\n');
    console.log('Error Code:', error.code);
    console.log('Error Message:', error.message);
    
    if (error.code === 'functions/not-found') {
      console.log('Function "predict_hemoglobin_integrated" does not exist!');
      console.log('   Make sure you deployed with the correct entry point.\n');
    } else if (error.message.includes('database URL') || error.message.includes('Database URL')) {
      console.log('Database URL not configured!');
      console.log('   But the function EXISTS - just needs database URL fix.\n');
    } else if (error.code === 'not-found') {
      console.log('Data record not found in database!');
      console.log('   Try a different record_id that exists in /data\n');
    } else {
      console.log('Full Error Object:');
      console.log(error);
    }
  }
}

testCallable();