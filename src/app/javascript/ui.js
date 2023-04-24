var form = document.getElementById("form");

form.addEventListener('submit', function(event) {
    event.preventDefault();
    const input = document.getElementById("prompt").nodeValue;
    const wrappedInput = {"instances":[input]}
    fetch('/infer', {
        method: 'POST',
        body: JSON.stringify(wrappedInput),
        headers: {
            "Content-Type": "application/json",
          },
        cache: "no-cache",
    }).then(function(response) {
        text = response.json()
        console.log(text);

        document.getElementById("response").innerText = text;
    });
});