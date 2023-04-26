var btn = document.getElementById("submit");

const urlParams = new URLSearchParams(location.search);
if (urlParams.has("fakedata")) {
    var fakeData=true
}

btn.addEventListener('click', function(event) {
    event.preventDefault();
    var input = document.getElementById("prompt").value;

    if (!document.getElementById("rawInputCheckbox").checked) {
        input = {"instances":[input]}
    }

    var responseElement = document.getElementById("response")
    responseJson = {}
    if (fakeData) {
        responseJson = {"predictions": ["'Tapi Tapi -- Handcrafted, authentic African ice cream' is a", "'Tapi Tapi -- Handcrafted, authentic African ice cream' is a", "'Tapi Tapi -- Handcrafted, authentic African ice cream' is a", "'Tapi Tapi -- Handcrafted, authentic African ice cream' is a", "'Tapi Tapi -- Handcrafted, authentic African ice cream' is a"]};
    }
    else {
        fetch('/infer', {
            method: 'POST',
            body: JSON.stringify(input),
            headers: {
                "Content-Type": "application/json",
            },
            cache: "no-cache",
        })
        .then((response) => response.json())
        .then(function(json) {
            responseJson = json;
        });
    }

    console.log(responseJson);
    predictions = responseJson["predictions"];
    if (predictions.length == 1) {
        predictions = predictions[0];
    }
    
    responseElement.value = JSON.stringify(predictions, null, 2);
    responseElement.dispatchEvent(new Event("input"))
});

$(function () {
    $('[data-toggle="tooltip"]').tooltip()
  })