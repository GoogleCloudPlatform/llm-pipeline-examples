var btn = document.getElementById("submit");

const urlParams = new URLSearchParams(location.search);
if (urlParams.has("fakedata")) {
    var fakeData=true
}

btn.addEventListener('click', async function(event) {
    event.preventDefault();
    var input = document.getElementById("prompt").value;

    if (!document.getElementById("rawInputCheckbox").checked) {
        input = {"instances":[input]}
    }

    var responseElement = document.getElementById("response")
    var responsePromise = {}
    if (fakeData) {
        responsePromise = Promise.resolve({"metrics": {"preprocessing": 10, "inferencing": 30, "postprocessing": 2}, "predictions": ["'Tapi Tapi -- Handcrafted, authentic African ice cream' is a", "'Tapi Tapi -- Handcrafted, authentic African ice cream' is a", "'Tapi Tapi -- Handcrafted, authentic African ice cream' is a", "'Tapi Tapi -- Handcrafted, authentic African ice cream' is a", "'Tapi Tapi -- Handcrafted, authentic African ice cream' is a"]});
    }
    else {
        responsePromise = fetch('/infer?metrics=true', {
            method: 'POST',
            body: JSON.stringify(input),
            headers: {
                "Content-Type": "application/json",
            },
            cache: "no-cache",
        })
        .then((response) => response.json());
    }

    var responseJson = await responsePromise;
    console.log(responseJson);
    var predictions = responseJson["predictions"];
    if (predictions.length == 1) {
        predictions = predictions[0];
    }

    if (responseJson["metrics"] != null) {
        var metricsElement = document.getElementById("response-metrics")
        metricsElement.innerHTML = JSON.stringify(responseJson["metrics"], null, 2);
    }
    
    responseElement.value = JSON.stringify(predictions, null, 2);
    responseElement.dispatchEvent(new Event("input"))
});

$(function () {
    $('[data-toggle="tooltip"]').tooltip()
  })