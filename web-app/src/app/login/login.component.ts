import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { ClientService, Server } from '../grpc/client.service';



@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.sass']
})
export class LoginComponent implements OnInit {

  submitted = false;

  constructor(public clientService: ClientService,
    private router: Router) {
  }

  ngOnInit(): void {
    this.clientService.load()
    this.clientService.connect()
      .then(result => {
        console.log('Connect result:', result);
        // this.router.navigate(['/camera']);
      })
      .catch((error) => console.log(error));
  }

  onSubmit(): void {
    this.submitted = true;
    console.log('Connect:', this.clientService.server);
    this.clientService.connect()
      .then(result => {
        console.log('Connect result:', result);
        this.router.navigate(['/camera']);
      })
      .catch((error) => alert(error))
      .finally(() => this.submitted = false);
  }
  onChange(event: Event):void{
    this.clientService.server.passwordChanged = true;
    console.log('Password changed.');
  }
}
