// JavaScript for uploaded.html
document.addEventListener("DOMContentLoaded", function () {
    const resultItems = document.querySelectorAll(".result-item");
    resultItems.forEach((item) => {
        item.addEventListener("click", function () {
            alert(`You clicked on: ${item.textContent}`);
        });
    });
});
