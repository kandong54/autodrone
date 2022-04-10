import { Component, OnInit } from '@angular/core';
import { ClientService, Server } from '../grpc/client.service';



@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.sass']
})
export class LoginComponent implements OnInit {
  server: Server = {
    protocol: 'https://',
    address: 'drone.kandong.dev',
    port: 10000,
    password: 'password'
  };
  submitted = false;
  constructor(private clientService: ClientService) {
  }

  ngOnInit(): void {
  }

  onSubmit(): void {

    console.log('Connect drone:', this.server);
    this.submitted = true;
    this.clientService.connect(this.server)
      .then(result => {
        console.log('Connect result:', result);
      })
      .catch((error) => alert(error))
      .finally(() => this.submitted = false);
  }

}
