// Simulate hospital responses and navigation
document.getElementById('patient-form').addEventListener('submit', function(event) {
  event.preventDefault();

  // Clear previous responses
  const hospitalList = document.getElementById('hospital-list');
  hospitalList.innerHTML = '';

  // Simulate the process of sending details to hospitals
  const hospitals = [
    { name: 'City Hospital', distance: 5, accepted: true },
    { name: 'Green Valley Medical', distance: 3, accepted: true },
    { name: 'Lakeside Healthcare', distance: 7, accepted: false },
    { name: 'Sunshine Hospital', distance: 4, accepted: true },
    { name: 'Riverfront Clinic', distance: 2, accepted: false }
  ];

  // Process hospital responses
  hospitals.forEach(hospital => {
    const listItem = document.createElement('li');
    if (hospital.accepted) {
      listItem.innerText = `${hospital.name} - Distance: ${hospital.distance} km - Accepted`;
    } else {
      listItem.innerText = `${hospital.name} - Distance: ${hospital.distance} km - Declined`;
    }
    hospitalList.appendChild(listItem);
  });

  // Find the nearest hospital that accepted
  const acceptedHospitals = hospitals.filter(h => h.accepted);
  const nearestHospital = acceptedHospitals.reduce((prev, curr) => {
    return prev.distance < curr.distance ? prev : curr;
  });

  // Display navigation result
  const navigationResult = document.getElementById('navigation-result');
  navigationResult.innerText = `Navigate to ${nearestHospital.name}, which is ${nearestHospital.distance} km away.`;
});
