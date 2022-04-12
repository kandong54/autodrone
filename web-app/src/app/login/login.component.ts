import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { ClientService, Server } from '../grpc/client.service';



@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.sass']
})
export class LoginComponent implements OnInit {
  server: Server = {
    protocol: '',
    address: '',
    port: 0,
    password: '',
    passwordHashed: ''
  };
  submitted = false;
  constructor(private clientService: ClientService,
    private router: Router) {
  }

  ngOnInit(): void {
    this.clientService.load()
    this.server = this.clientService.server;
    this.clientService.connect()
      .then(result => {
        console.log('Connect result:', result);
        this.router.navigate(['/camera']);
      })
      .catch((error) => console.log(error));
  }

  onSubmit(): void {
    this.submitted = true;
    console.log('Connect:', this.server);
    this.clientService.connect(this.server)
      .then(result => {
        console.log('Connect result:', result);
        this.router.navigate(['/camera']);
      })
      .catch((error) => alert(error))
      .finally(() => this.submitted = false);
  }

}
