function sendUserMessage() {
    var rawText = $("#textInput").val();
    if (rawText.trim() === "") return;

    // 1. Show user message
    var userHtml = '<div class="user-msg"><p>' + rawText + '</p></div>';
    $("#textInput").val("");
    $("#chatbox").append(userHtml);
    
    // Scroll to bottom
    document.getElementById("chatbox").scrollTop = document.getElementById("chatbox").scrollHeight;

    // 2. Send to Flask Backend
    $.ajax({
        data: JSON.stringify({ message: rawText }), 
        contentType: "application/json",            
        type: "POST",
        url: "/chat",
    }).done(function(data) {
        // 3. Show Bot Response
        var botHtml = '<div class="bot-msg"><p>' + data.response + '</p></div>';
        $("#chatbox").append(botHtml);
        document.getElementById("chatbox").scrollTop = document.getElementById("chatbox").scrollHeight;
    });
}

// Allow "Enter" key to submit
$("#textInput").keypress(function(e) {
    if(e.which == 13) {
        sendUserMessage();
    }
});