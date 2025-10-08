function f() {
    let sum = 0;
    for (let i = 0; i < 1e7; i++) {  // 2 billion iterations
        sum += i % 10;
    }
    console.log("Sum:", sum);
}

f();

