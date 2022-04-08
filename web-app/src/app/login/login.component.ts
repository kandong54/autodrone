import { Component, OnInit } from '@angular/core';

class Hostname {
  protocol: string = "https://";
  address: string = "127.0.0.1";
  port: number = 10000;
  password: string = "admin";

  toString(): string {
    return `${this.protocol}${this.address}:${this.port}@${this.password}`;
  }
}

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.sass']
})
export class LoginComponent implements OnInit {

  hostname = new Hostname();

  constructor() { }

  ngOnInit(): void {
  }

  onSubmit(): void {
    console.log("Connect drone:", this.hostname.toString());
  }

}
