clear all;clc
disp('Case 1 : cart-pole');
disp('Case 2 : one-tank'); 
n = input('Enter a number that define your plant: ');
switch n 
    case 1
        disp('cart-pole system');
        plant_cart_pole;
    case 2
        disp('one-tank system')
        plant_one_tank;
end