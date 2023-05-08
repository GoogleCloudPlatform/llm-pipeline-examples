var btn = document.getElementById("submit");

btn.addEventListener('click', async function(event) {
    event.preventDefault();
    var input = document.getElementById("prompt").value;

    if (!document.getElementById("rawInputCheckbox").checked) {
        input = {"instances":[input]}
    }

    var responseElement = document.getElementById("response")
    var responsePromise = fetch('/infer?metrics=true', {
        method: 'POST',
        body: JSON.stringify(input),
        headers: {
            "Content-Type": "application/json",
        },
        cache: "no-cache",
    })
    .then((response) => response.json());

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